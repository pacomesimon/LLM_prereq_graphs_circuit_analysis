import networkx as nx
import time
import json
import io
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from datetime import datetime

# --- Structured Output for Evaluator ---
class EvalResponse(BaseModel):
    """The model's selected answer ID."""
    choice_id: int = Field(description="The index of the correct choice (0, 1, 2, or 3).")

# --- Evaluation Engine ---
class EvaluationEngine:
    def __init__(self, models: List[str], failure_threshold: int, provider: str = "OpenAI", api_key: Optional[str] = None):
        """
        models: List of model names sorted from smallest to largest capability.
        failure_threshold: Number of allowed failures along a path before upgrading the model.
        provider: 'OpenAI' or 'Ollama'.
        api_key: Optional OpenAI API key.
        """
        self.models = models
        self.threshold = failure_threshold
        self.provider = provider
        self.api_key = api_key
        self.results_map = {} # To store node-specific results for reporting

    def run_cascade_eval(self, df: pd.DataFrame):
        # 1. Build Directed Acyclic Graph (DAG)
        G = nx.DiGraph()
        node_data = {}
        for _, row in df.iterrows():
            node_id = str(row['id'])
            G.add_node(node_id)
            node_data[node_id] = row.to_dict()
            if row['parent_id'] and str(row['parent_id']) != "None":
                G.add_edge(str(row['parent_id']), node_id)

        # 2. Identify Root Nodes
        roots = [n for n, d in G.in_degree() if d == 0]
        
        # 3. DFS Traversal with Stack State
        # Stack structure: (node_id, current_model_idx, path_failures, path_successes, is_skipped)
        stack = []
        for root in reversed(roots): # Process roots
            stack.append((root, 0, 0, 0, False))

        visited = set()
        final_results = []

        while stack:
            node_id, model_idx, p_fail, p_succ, is_skipped = stack.pop()
            
            if node_id in visited:
                continue
            visited.add(node_id)

            # Record for this node
            node_info = node_data[node_id]
            result_entry = {
                "id": node_id,
                "parent_id": node_info.get('parent_id'),
                "tags": node_info.get('tags'),
                "ground_truth": node_info.get('correct_idx')
            }

            # Handle Propagation of Failure
            if is_skipped:
                result_entry.update({
                    "used_model": "failed_all (skipped)",
                    "prediction": -1,
                    "path_successes": p_succ,
                    "path_failures": p_fail,
                    "inference_time": 0,
                    "status": "skipped"
                })
                self.results_map[node_id] = result_entry
                final_results.append(result_entry)
                
                # Push children to stack as skipped
                for child in reversed(list(G.successors(node_id))):
                    stack.append((child, model_idx, p_fail, p_succ, True))
                continue

            # --- MODEL EVALUATION / UPGRADE LOOP ---
            success = False
            curr_idx = model_idx
            
            while curr_idx < len(self.models):
                model_name = self.models[curr_idx]
                start_time = time.time()
                
                try:
                    prediction = self._call_model(model_name, node_info)
                    inf_time = time.time() - start_time
                    
                    if prediction == int(node_info['correct_idx']):
                        # SUCCESS
                        result_entry.update({
                            "used_model": model_name,
                            "prediction": prediction,
                            "path_successes": p_succ + 1,
                            "path_failures": p_fail,
                            "inference_time": inf_time,
                            "status": "success"
                        })
                        success = True
                        break
                    else:
                        # FAILURE (Wrong Answer)
                        p_fail += 1
                        if p_fail > self.threshold:
                            # Upgrade search: try next model on same question
                            curr_idx += 1
                            continue # Loop to next model
                        else:
                            # Failure recorded but stay with current model for now
                            result_entry.update({
                                "used_model": model_name,
                                "prediction": prediction,
                                "path_successes": p_succ,
                                "path_failures": p_fail,
                                "inference_time": inf_time,
                                "status": "fail"
                            })
                            break # Move to children
                except Exception as e:
                    # API EXCEPTION counts as failure
                    print(f"Error calling {model_name}: {e}")
                    curr_idx += 1
                    p_fail += 1
            
            if not success and curr_idx >= len(self.models):
                # ALL MODELS FAILED
                result_entry.update({
                    "used_model": "failed_all",
                    "prediction": -1,
                    "path_successes": p_succ,
                    "path_failures": p_fail,
                    "inference_time": 0,
                    "status": "critical_failure"
                })
                is_skipped_for_children = True
            else:
                is_skipped_for_children = False

            self.results_map[node_id] = result_entry
            final_results.append(result_entry)

            # 4. Push children with persistent state
            # If successful, we use the potentially 'upgraded' model for all descendants
            for child in reversed(list(G.successors(node_id))):
                stack.append((child, curr_idx, p_fail, p_succ + (1 if success else 0), is_skipped_for_children))

        return pd.DataFrame(final_results)

    def _call_model(self, model_name: str, node_info: Dict) -> int:
        """Invokes chosen model with tool-calling for the answer ID."""
        if self.provider == "Ollama":
            llm = ChatOllama(model=model_name, temperature=0, format="json").with_structured_output(EvalResponse)
        else:
            llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=self.api_key).with_structured_output(EvalResponse)
        
        prompt = (
            f"Question: {node_info['question']}\n\n"
            f"Choices:\n{node_info['choices']}\n\n"
            "Identify the correct choice index (0, 1, 2, or 3). Respond with JSON matching the required schema."
        )
        
        res = llm.invoke(prompt)
        return int(res.choice_id)

    def generate_visualization(self, results_df):
        """Generates the NetworkX graph plot using a radial shell layout."""
        G = nx.DiGraph()
        color_map = {}
        labels = {}
        
        # 1. Build the graph for positioning
        for _, row in results_df.iterrows():
            node_id = str(row['id'])
            G.add_node(node_id)
            if row['parent_id'] and str(row['parent_id']) != "None":
                G.add_edge(str(row['parent_id']), node_id)
            
            status = row.get('status', 'success')
            color_map[node_id] = 'lightcoral' if status in ['fail', 'critical_failure', 'skipped'] else 'lightblue'
            
            labels[node_id] = (
                f"ID: {node_id}\n"
                f"M: {row['used_model']}"
            )

        colors = [color_map.get(node, 'lightgray') for node in G.nodes()]

        # 2. Calculate shells based on depth
        roots = [n for n, d in G.in_degree() if d == 0]
        shells = [roots]
        
        # Iterate to find subsequent layers
        current_layer = roots
        visited = set(roots)
        while True:
            next_layer = []
            for node in current_layer:
                for neighbor in G.successors(node):
                    if neighbor not in visited:
                        next_layer.append(neighbor)
                        visited.add(neighbor)
            if not next_layer:
                break
            shells.append(next_layer)
            current_layer = next_layer

        # 3. Apply shell layout
        pos = nx.shell_layout(G, shells)

        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, labels=labels, 
                node_color=colors, node_size=2500, font_size=8, 
                font_weight='bold', font_color='black', 
                edge_color='gray', arrows=True, arrowsize=15)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        return buf.read()

    def generate_summary(self, df: pd.DataFrame) -> str:
        """Generates textual summary report including failed/skipped nodes."""
        summary = "### 📊 Compression Strategy Summary\n\n"
        
        # Aggregate stats for ALL outcomes (including failures/skips)
        stats = df.groupby('used_model').agg({
            'id': 'count',
            'status': lambda x: (x == 'success').sum(),
            'inference_time': 'mean'
        }).rename(columns={'id': 'Total Count', 'status': 'Successes', 'inference_time': 'Avg Time (s)'})
        
        summary += stats.to_markdown() + "\n\n"
        
        total_q = len(df)
        total_s = (df['status'] == 'success').sum()
        summary += f"**Overall Accuracy:** {total_s/total_q:.1%} ({total_s}/{total_q})"
        
        return summary

    def get_tag_sets(self, df: pd.DataFrame, apply_monotonic: bool = False) -> Dict[str, set]:
        """
        Aggregates unique tags. 
        If apply_monotonic is True: 
        - Pass tags of smaller models are inherited by larger ones.
        - Failure sets are defined by the limits of the largest model.
        """
        tag_sets = {}
        
        def parse_tags(t):
            if isinstance(t, list): return set(t)
            try: return set(json.loads(t))
            except: return set()

        # Identify models in ascending order of capability (based on appearance order)
        all_used_models = df['used_model'].unique().tolist()
        ordered_models = [m for m in all_used_models if m not in ["failed_all", "skipped"] and "skipped" not in m]
        
        # 1. Per-model Pass/Fail sets
        model_pass_data = {}
        model_fail_data = {}

        for model in ordered_models:
            m_df = df[df['used_model'] == model]
            pass_tags = set().union(*(m_df[m_df['status'] == 'success']['tags'].apply(parse_tags)))
            fail_tags = set().union(*(m_df[m_df['status'] == 'fail']['tags'].apply(parse_tags)))
            
            model_pass_data[model] = pass_tags
            model_fail_data[model] = fail_tags

        if apply_monotonic:
            # Cumulative union: Model N inherits all tags from Model N-1
            accumulated_pass = set()
            for model in ordered_models:
                accumulated_pass = accumulated_pass.union(model_pass_data[model])
                tag_sets[f"{model} (Pass)"] = set(accumulated_pass)
                # For monotonic fail, we only care about what the model actually failed
                tag_sets[f"{model} (Fail)"] = model_fail_data[model]
        else:
            for model in ordered_models:
                if model_pass_data[model]: tag_sets[f"{model} (Pass)"] = model_pass_data[model]
                if model_fail_data[model]: tag_sets[f"{model} (Fail)"] = model_fail_data[model]

        # 2. Global Failure sets
        if apply_monotonic and ordered_models:
            # Largest model defines the baseline of 'Failed All'
            largest_model = ordered_models[-1]
            f_all_tags = tag_sets.get(f"{largest_model} (Fail)", set())
            f_all_tags |= tag_sets.get(f"{largest_model} (Pass)", set())
            failed_all_df = df[df['used_model'] == "failed_all"]
            f_all_tags |= set().union(*(failed_all_df['tags'].apply(parse_tags)))
            tag_sets["Failed All"] = f_all_tags
            
            # 'Skipped' inherits everything from 'Failed All' (representing extreme complexity)
            skipped_df = df[df['used_model'].str.contains("skipped", na=False)]
            s_tags = set().union(*(skipped_df['tags'].apply(parse_tags)))
            tag_sets["Skipped"] = s_tags.union(f_all_tags)
        else:
            failed_all_df = df[df['used_model'] == "failed_all"]
            skipped_df = df[df['used_model'].str.contains("skipped", na=False)]
            tag_sets["Failed All"] = set().union(*(failed_all_df['tags'].apply(parse_tags)))
            tag_sets["Skipped"] = set().union(*(skipped_df['tags'].apply(parse_tags)))
        
        return tag_sets

    def plot_tag_venn(self, tag_sets: Dict[str, set], selected_names: List[str]):
        """Generates a Venn diagram for selected tag sets with a separate legend."""
        from matplotlib_venn import venn2, venn3
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        
        if len(selected_names) < 2:
            return None, "Select at least 2 sets to compare."
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sets = [tag_sets[name] for name in selected_names]
        
        # 1. Create the Venn without labels over the circles
        if len(selected_names) == 2:
            v = venn2(sets, set_labels=["", ""])
        else:
            v = venn3(sets[:3], set_labels=["", "", ""])
            
        # 2. Extract colors for the legend
        # Primary circle IDs: '10' and '01' for 2 sets; '100', '010', '001' for 3 sets
        ids = ['10', '01'] if len(selected_names) == 2 else ['100', '010', '001']
        legend_elements = []
        
        for i, patch_id in enumerate(ids):
            patch = v.get_patch_by_id(patch_id)
            color = patch.get_facecolor() if patch else "gray"
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=selected_names[i],
                                          markerfacecolor=color, markersize=15))
        
        plt.legend(handles=legend_elements, title="Tag Sets", loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f"Tag Set Intersections: {', '.join(selected_names)}", pad=20)
        
        # Capture plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close()
        buf.seek(0)
        
        # Also return textual intersections for UI display
        intersections_report = f"### 🏷️ Tag Intersections Details ({', '.join(selected_names)})\n\n"
        if len(selected_names) == 2:
            s1, s2 = sets[0], sets[1]
            intersections_report += f"**Only {selected_names[0]}:** {sorted(list(s1 - s2))}\n\n"
            intersections_report += f"**Overlap:** {sorted(list(s1 & s2))}\n\n"
            intersections_report += f"**Only {selected_names[1]}:** {sorted(list(s2 - s1))}\n\n"
        elif len(selected_names) == 3:
            s1, s2, s3 = sets[0], sets[1], sets[2]
            intersections_report += f"**Triple Overlap ({selected_names[0]} & {selected_names[1]} & {selected_names[2]}):** {sorted(list(s1 & s2 & s3))}\n\n"
            intersections_report += f"**Shared ({selected_names[0]} & {selected_names[1]} but NOT {selected_names[2]}):** {sorted(list((s1 & s2) - s3))}\n\n"
            intersections_report += f"**Shared ({selected_names[0]} & {selected_names[2]} but NOT {selected_names[1]}):** {sorted(list((s1 & s3) - s2))}\n\n"
            intersections_report += f"**Shared ({selected_names[1]} & {selected_names[2]} but NOT {selected_names[0]}):** {sorted(list((s2 & s3) - s1))}\n\n"
            intersections_report += f"**Only {selected_names[0]}:** {sorted(list(s1 - s2 - s3))}\n\n"
            intersections_report += f"**Only {selected_names[1]}:** {sorted(list(s2 - s1 - s3))}\n\n"
            intersections_report += f"**Only {selected_names[2]}:** {sorted(list(s3 - s1 - s2))}\n\n"
        
        return buf.read(), intersections_report
