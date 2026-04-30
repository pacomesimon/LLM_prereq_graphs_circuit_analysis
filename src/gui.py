import gradio as gr
import pandas as pd
import json
import os
from .dataset_logic import DatasetAgent, save_to_json, load_from_json

# Global instance for UI
agent = DatasetAgent()

def generate_blueprint_handler(topic, depth, breadth, model_name, api_key):
    if not topic:
        return None, None
    df = agent.generate_blueprint(topic, int(depth), int(breadth), model_name=model_name, api_key=api_key or None)
    filepath = save_to_json(df, "blueprint")
    return df, filepath

def generate_qa_handler(blueprint_df, model_name, api_key):
    if blueprint_df is None or blueprint_df.empty:
        return None, None
    # Agent iterates over the tags in the blueprint to generate MCQs
    final_df = agent.generate_qa_batch(blueprint_df, model_name=model_name, api_key=api_key or None)
    filepath = save_to_json(final_df, "full_dataset")
    return final_df, filepath

def load_json_handler(file):
    if file is None: return None
    return load_from_json(file)

def save_json_handler(df, prefix):
    if df is None: return None
    return save_to_json(df, prefix)

def create_ui():
    """
    Constructs the Performance-Aware Model Compression for Circuit Analysis GUI.
    """
    theme = gr.themes.Soft(
        primary_hue="blue", 
        secondary_hue="gray",
        font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
    )
    
    with gr.Blocks(theme=theme, title="Model Compression Engine") as app:
        gr.Markdown("""
        # 🧪 Performance-Aware Model Compression for Circuit Analysis
        *A workflow for building hierarchical prerequisite graphs for model compression evaluation.*
        """)
        
        with gr.Accordion("🔑 API Configuration", open=False):
            api_key_input = gr.Textbox(
                label="OpenAI API Key (Optional, defaults to .env)",
                placeholder="sk-...",
                type="password"
            )
        
        with gr.Tabs():
            # PHASE 1: BLUEPRINTTab
            with gr.TabItem("Phase 1: Dataset Blueprint"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Configuration")
                        model_dropdown = gr.Dropdown(
                            choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                            value="gpt-4o",
                            label="OpenAI Model for Generation"
                        )
                        topic = gr.Textbox(
                            label="Main Subject / Topic", 
                            placeholder="e.g. Modern Web Infrastructure and Scalability",
                            lines=2
                        )
                        with gr.Row():
                            depth = gr.Slider(1, 5, 2, step=1, label="Graph Depth")
                            breadth = gr.Slider(1, 4, 2, step=1, label="Parent Fan-out")
                        
                        gen_bp_btn = gr.Button("🚀 Generate Blueprint", variant="primary")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 2. File Import/Export")
                        upload_bp = gr.File(
                            label="Upload Existing Blueprint (JSON)", 
                            file_types=[".json"]
                        )
                        gr.Examples(
                            examples=[["assets/blueprint_7o054luj.json"]],
                            inputs=upload_bp,
                            label="Blueprint Example"
                        )
                        download_bp_link = gr.File(label="Download Generated Blueprint", interactive=False)

                    with gr.Column(scale=3):
                        gr.Markdown("### 3. Blueprint Editor (Hierarchical Tags)")
                        blueprint_table = gr.Dataframe(
                            interactive=True,
                            wrap=True,
                            label="Edit Tags and Structure before Phase 2"
                        )
                
                gen_bp_btn.click(
                    generate_blueprint_handler, 
                    inputs=[topic, depth, breadth, model_dropdown, api_key_input], 
                    outputs=[blueprint_table, download_bp_link]
                )
                
                upload_bp.change(load_json_handler, inputs=[upload_bp], outputs=[blueprint_table])

            # PHASE 2: Q&A GENERATION
            with gr.TabItem("Phase 2: Question & Answer Generation"):
                gr.Markdown("### Submit the Blueprint for Full Q&A Generation")
                gr.Info("The agent will iterate over every entry's tags and generate specific MCQs using function calling.")
                
                with gr.Row():
                    run_qa_btn = gr.Button("⚡ Generate Questions & Answers", variant="primary", scale=1)
                    save_final_btn = gr.Button("💾 Save Final Dataset Snapshot", variant="secondary", scale=1)

                with gr.Row():
                    with gr.Column(scale=1):
                        upload_final = gr.File(
                            label="Upload Final Dataset (JSON)", 
                            file_types=[".json"]
                        )
                        gr.Examples(
                            examples=[["assets/full_dataset_6hxx81y3.json"]],
                            inputs=upload_final,
                            label="Full Dataset Example"
                        )
                        download_final_link = gr.File(label="Download Full Results", interactive=False)
                        
                    with gr.Column(scale=3):
                        final_table = gr.Dataframe(
                            interactive=True,
                            wrap=True,
                            label="Final Dataset Preview (MCQ & Ground Truth)"
                        )

                run_qa_btn.click(
                    generate_qa_handler,
                    inputs=[blueprint_table, model_dropdown, api_key_input],
                    outputs=[final_table, download_final_link]
                )
                
                save_final_btn.click(
                    save_json_handler,
                    inputs=[final_table, gr.State("final_dataset")],
                    outputs=[download_final_link]
                )
                
                upload_final.change(load_json_handler, inputs=[upload_final], outputs=[final_table])

            # PHASE 3: STRATEGIC EVALUATION
            with gr.TabItem("3. Strategic Evaluation"):
                tag_sets_state = gr.State({}) # Stores computed sets
                full_results_state = gr.State(None) # Stores raw DF for re-calc
                gr.Markdown("### 🪜 Hierarchical Model Cascade & DFS Eval")
                gr.Info("""
                This phase runs a 'laddering' evaluation:
                -   Starts with the smallest model.
                -   Tracks **path-cumulative failures**.
                -   **Upgrades** to a larger model if failures exceed the threshold.
                -   Perpetuates the better model for all child descendants.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("#### Evaluation Config")
                            provider_selection = gr.Radio(
                                choices=["OpenAI", "Ollama"],
                                value="Ollama",
                                label="Model Provider"
                            )
                            model_list = gr.Textbox(
                                label="Sorted Model List (Smallest to Largest)", 
                                value="gemma3:270m, gemma3:1b, gemma3:4b",
                                placeholder="e.g. gpt-4o-mini, gpt-4o"
                            )
                            model_list_examples = gr.Examples(
                                examples=[["gpt-4o-mini, gpt-4o"],
                                          ["gemma3:270m, gemma3:1b, gemma3:4b"]],
                                inputs=model_list,
                                label="Model List Example"
                            )
                            fail_threshold = gr.Slider(0, 5, 1, step=1, label="Path Failure Threshold (Upgrade Trigger)")
                            
                        run_eval_btn = gr.Button("🔎 Start Strategic Evaluation", variant="primary")
                        
                        gr.Markdown("---")
                        gr.Markdown("#### Output Summary")
                        eval_summary_out = gr.Markdown("No evaluation run yet.")
                        eval_download_link = gr.File(label="Download Final Report", interactive=False)
                        

                    with gr.Column(scale=3):
                        gr.Markdown("---")
                        upload_eval_report = gr.File(label="Upload Existing Eval Report (JSON)", file_types=[".json"])
                        with gr.Row():
                            eval_report_example_file_path = "assets/full_eval_report_2u970ry9.json"
                            gr.Examples(
                                examples=[eval_report_example_file_path],
                                inputs=upload_eval_report,
                                label="Eval Report Example"
                            )
                            example_btn_uploader = gr.Button("Load Example")
                            example_btn_uploader.click(
                                fn = lambda: eval_report_example_file_path,
                                inputs=[],
                                outputs=upload_eval_report
                            )
                        with gr.Tabs():
                            with gr.TabItem("📈 Graph Visualization"):
                                eval_viz_out = gr.Image(label="Dependency & Model Usage Map", type="numpy")
                            with gr.TabItem("🏷️ Tag Set Analysis"):
                                with gr.Row():
                                    set_selector = gr.Dropdown(
                                        choices=[], 
                                        multiselect=True, 
                                        max_choices=3, 
                                        label="Select Sets to Compare (Min 2, Max 3)"
                                    )
                                    update_venn_btn = gr.Button("🔄 Update Venn Diagram")
                                with gr.Row():
                                    monotonic_checkbox = gr.Checkbox(
                                        label="Apply Monotonic Capability Assumption (Small → Large inheritance)",
                                        value=False
                                    )
                                with gr.Row():
                                    venn_img_out = gr.Image(label="Venn Diagram", type="numpy")
                                    venn_report_out = gr.Markdown("Select sets and click update.")
                            with gr.TabItem("📋 Full Results Table"):
                                eval_table_out = gr.Dataframe(interactive=False, wrap=False)
                        

                def run_strategic_eval_handler(df, provider, models_str, threshold, api_key):
                    from .eval_logic import EvaluationEngine
                    if df is None or df.empty or not models_str:
                        return "Please provide dataset.", None, None, None, {}, gr.update()
                    
                    # 1. Parse Models
                    models = [m.strip() for m in models_str.split(',')]
                    
                    # 2. Run Engine
                    engine = EvaluationEngine(models, int(threshold), provider=provider, api_key=api_key or None)
                    results_df = engine.run_cascade_eval(df)
                    
                    # 3. Finalize results
                    summary = engine.generate_summary(results_df)
                    viz_img = engine.generate_visualization(results_df)
                    report_path = save_to_json(results_df, "full_eval_report")
                    tag_sets = engine.get_tag_sets(results_df)
                    available_sets = list(tag_sets.keys())
                    
                    import numpy as np
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(viz_img))
                    
                    return summary, report_path, np.array(img), results_df, tag_sets, gr.update(choices=available_sets, value=available_sets[:3] if len(available_sets) >=3 else available_sets)

                def load_eval_report_handler(file):
                    from .eval_logic import EvaluationEngine # load_from_json is already imported in this file and is not available in .eval_logic
                    if file is None: return None, None, None, None, {}, gr.update(choices=[], value=[])
                    
                    results_df = load_from_json(file)
                    if results_df.empty:
                        return "Error loading file.", None, None, None, {}, gr.update(choices=[], value=[])
                    
                    # Instantiate engine with defaults just for viz/summary mapping
                    engine = EvaluationEngine([], 0) 
                    summary = engine.generate_summary(results_df)
                    viz_img = engine.generate_visualization(results_df)
                    tag_sets = engine.get_tag_sets(results_df)
                    available_sets = list(tag_sets.keys())
                    
                    import numpy as np
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(viz_img))
                    
                    return summary, file.name, np.array(img), results_df, tag_sets, gr.update(choices=available_sets, value=available_sets[:3] if len(available_sets) >=3 else available_sets)

                def update_venn_handler(results_df, selected, apply_monotonic):
                    from .eval_logic import EvaluationEngine
                    if results_df is None or len(selected) < 2:
                        return None, "Select at least 2 sets."
                    
                    engine = EvaluationEngine([], 0)
                    tag_sets = engine.get_tag_sets(results_df, apply_monotonic=apply_monotonic)
                    viz_bytes, report = engine.plot_tag_venn(tag_sets, selected)
                    
                    import numpy as np
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(viz_bytes))
                    return np.array(img), report

                run_eval_btn.click(
                    run_strategic_eval_handler,
                    inputs=[final_table, provider_selection, model_list, fail_threshold, api_key_input],
                    outputs=[eval_summary_out, eval_download_link, eval_viz_out, eval_table_out, tag_sets_state, set_selector]
                ).then(
                    lambda df: df, inputs=[eval_table_out], outputs=[full_results_state]
                )
                
                upload_eval_report.change(
                    load_eval_report_handler,
                    inputs=[upload_eval_report],
                    outputs=[eval_summary_out, eval_download_link, eval_viz_out, eval_table_out, tag_sets_state, set_selector]
                ).then(
                    lambda df: df, inputs=[eval_table_out], outputs=[full_results_state]
                )

                update_venn_btn.click(
                    update_venn_handler,
                    inputs=[full_results_state, set_selector, monotonic_checkbox],
                    outputs=[venn_img_out, venn_report_out]
                )

    return app
