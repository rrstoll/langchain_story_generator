# app.py - Fixed for complete responses without truncation
import gradio as gr
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Get API key from Hugging Face Secrets
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    def no_api_key_interface():
        return """
        ⚠️ **OpenAI API Key Required**
        
        To use this app, add an OpenAI API key:
        1. Go to Space Settings → Repository secrets
        2. Add secret: `OPENAI_API_KEY`
        3. Paste your OpenAI API key
        4. Restart the space
        """
    
    demo = gr.Interface(
        fn=lambda x: no_api_key_interface(),
        inputs=gr.Textbox(label="Story Idea", placeholder="API key required"),
        outputs=gr.Markdown(),
        title="🎬 AI Story Generator - Setup Required"
    )

else:
    def create_story_generator():
        """Create story generation with individual LLM instances for better token control"""
        output_parser = StrOutputParser()
        
        # Create separate LLM instances with different token limits for each step
        concept_llm = OpenAI(
            model="gpt-3.5-turbo-instruct",
            temperature=0.7,
            max_tokens=400,  # Small, focused output
            openai_api_key=api_key,
            request_timeout=60
        )
        
        plot_llm = OpenAI(
            model="gpt-3.5-turbo-instruct", 
            temperature=0.7,
            max_tokens=500,  # Medium output
            openai_api_key=api_key,
            request_timeout=60
        )
        
        character_llm = OpenAI(
            model="gpt-3.5-turbo-instruct",
            temperature=0.7,
            max_tokens=600,  # Larger for character details
            openai_api_key=api_key,
            request_timeout=60
        )
        
        scene_llm = OpenAI(
            model="gpt-3.5-turbo-instruct",
            temperature=0.7,
            max_tokens=800,  # Largest for descriptive scene
            openai_api_key=api_key,
            request_timeout=60
        )
        
        pitch_llm = OpenAI(
            model="gpt-3.5-turbo-instruct",
            temperature=0.7,
            max_tokens=400,  # Concise pitch
            openai_api_key=api_key,
            request_timeout=60
        )
        
        # Very concise prompts to maximize output space
        concept_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""Idea: {user_input}

Create story concept:
Character: [name, age, key trait]
Setting: [when, where]
Conflict: [main problem]
Genre: [type]

Write complete concept:"""
        )
        
        plot_prompt = PromptTemplate(
            input_variables=["story_concept"],
            template="""Concept: {story_concept}

5-act plot:
Act 1: [setup - 2 sentences]
Act 2: [rising action - 2 sentences]
Act 3: [climax - 2 sentences]
Act 4: [falling action - 2 sentences]
Act 5: [resolution - 2 sentences]

Complete plot:"""
        )
        
        character_prompt = PromptTemplate(
            input_variables=["story_concept"],
            template="""Concept: {story_concept}

3 characters:
HERO: [name, age, personality, goal]
VILLAIN: [name, motivation, methods]
ALLY: [name, role, relationship]

Full profiles:"""
        )
        
        scene_prompt = PromptTemplate(
            input_variables=["story_concept", "characters"],
            template="""Concept: {story_concept}
Characters: {characters}

Opening scene (300 words):
Show hero in action, establish setting, hint at conflict.

Complete scene:"""
        )
        
        pitch_prompt = PromptTemplate(
            input_variables=["story_concept"],
            template="""Concept: {story_concept}

Marketing pitch:
Title: [catchy name]
Hook: [one sentence]
Summary: [3 sentences]
Audience: [who reads this]
Similar to: [comparable works]

Full pitch:"""
        )
        
        # Create chains with specific LLMs
        concept_chain = concept_prompt | concept_llm | output_parser
        plot_chain = plot_prompt | plot_llm | output_parser
        character_chain = character_prompt | character_llm | output_parser
        scene_chain = scene_prompt | scene_llm | output_parser
        pitch_chain = pitch_prompt | pitch_llm | output_parser
        
        def generate_complete_story(inputs):
            """Generate with careful token management"""
            try:
                user_input = inputs["user_input"]
                print(f"🎬 Generating: '{user_input[:30]}...'")
                
                # Step 1: Concept
                print("📝 Concept...")
                concept_result = concept_chain.invoke({"user_input": user_input})
                print(f"   ✅ {len(concept_result)} chars")
                
                # Step 2: Plot
                print("📋 Plot...")
                plot_result = plot_chain.invoke({"story_concept": concept_result})
                print(f"   ✅ {len(plot_result)} chars")
                
                # Step 3: Characters (only use concept to save tokens)
                print("👥 Characters...")
                character_result = character_chain.invoke({"story_concept": concept_result})
                print(f"   ✅ {len(character_result)} chars")
                
                # Step 4: Scene (use concept + short character summary)
                print("🎭 Scene...")
                # Use only first 200 chars of characters to save tokens
                short_characters = character_result[:200] + "..." if len(character_result) > 200 else character_result
                scene_result = scene_chain.invoke({
                    "story_concept": concept_result,
                    "characters": short_characters
                })
                print(f"   ✅ {len(scene_result)} chars")
                
                # Step 5: Pitch (only use concept)
                print("🎯 Pitch...")
                pitch_result = pitch_chain.invoke({"story_concept": concept_result})
                print(f"   ✅ {len(pitch_result)} chars")
                
                total_length = sum([
                    len(concept_result), len(plot_result), len(character_result),
                    len(scene_result), len(pitch_result)
                ])
                print(f"📊 Total: {total_length} characters")
                
                return {
                    "story_concept": concept_result,
                    "plot_outline": plot_result,
                    "characters": character_result,
                    "opening_scene": scene_result,
                    "marketing_pitch": pitch_result,
                    "total_length": total_length
                }
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                raise Exception(f"Failed: {str(e)}")
        
        return RunnableLambda(generate_complete_story)

    # Initialize story generator
    story_generator = create_story_generator()

    def generate_story(user_input, progress=gr.Progress()):
        """Main generation function"""
        
        if not user_input or not user_input.strip():
            return "💡 **Enter a story idea!**", "", "", "", ""
        
        if len(user_input.strip()) < 5:
            return "📝 **Too short!** Need at least 5 characters.", "", "", "", ""
        
        if len(user_input.strip()) > 200:
            return "✂️ **Too long!** Keep under 200 characters.", "", "", "", ""
        
        try:
            progress(0.1, desc="Starting...")
            result = story_generator.invoke({"user_input": user_input.strip()})
            progress(1.0, desc="Complete!")
            
            # Format with completion indicators
            concept = f"""## 📝 Story Concept
*{len(result['story_concept'])} characters*

{result['story_concept']}

---
✅ **Complete**"""

            plot = f"""## 📋 Plot Outline
*{len(result['plot_outline'])} characters*

{result['plot_outline']}

---
✅ **Complete**"""

            characters = f"""## 👥 Characters
*{len(result['characters'])} characters*

{result['characters']}

---
✅ **Complete**"""

            scene = f"""## 🎭 Opening Scene
*{len(result['opening_scene'])} characters*

{result['opening_scene']}

---
✅ **Complete**"""

            pitch = f"""## 🎯 Marketing Pitch
*{len(result['marketing_pitch'])} characters*

{result['marketing_pitch']}

---
✅ **Complete** | **Total:** {result['total_length']} characters"""
            
            return concept, plot, characters, scene, pitch
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "token" in error_str or "context" in error_str:
                error_msg = """🚫 **Still too complex!**

Try an even simpler idea:
- "Detective with superpowers"
- "Time travel romance"
- "Robot learns emotions"

Keep it to 3-5 words if possible."""
                
            elif "rate limit" in error_str:
                error_msg = "⏱️ **Rate limited.** Wait 2 minutes and try again."
                
            else:
                error_msg = f"❌ **Error:** {str(e)}\n\nTry a different idea."
            
            return error_msg, "", "", "", ""

    # Gradio interface
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="🎬 AI Story Generator"
    ) as demo:
        
        gr.Markdown("""
        # 🎬 AI Story Generator
        ### Simple ideas → Complete stories
        
        **Optimized for:** Complete, untruncated outputs
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                user_input = gr.Textbox(
                    label="💡 Your Story Idea",
                    placeholder="Detective with special powers",
                    lines=2,
                    max_lines=3,
                    info="Short and simple works best! (5-50 words)"
                )
                
                generate_btn = gr.Button(
                    "✨ Generate Story", 
                    variant="primary", 
                    size="lg"
                )
                
                gr.Markdown("""
                ### ✨ Perfect Examples:
                - Detective sees object memories
                - Time stops for librarians  
                - Plants use social media
                - Memory thief loses past
                - Lies become real
                - Invisible retail worker
                - Dream epidemic spreads
                - Robot learns to love
                - Magic coffee shop
                - Vampire dentist
                """)
        
        with gr.Column(scale=2):
            story_output = gr.Markdown("Enter a simple idea and generate your story!")
            
            with gr.Tabs():
                with gr.TabItem("📋 Plot"):
                    plot_tab = gr.Markdown()
                with gr.TabItem("👥 Characters"):
                    char_tab = gr.Markdown()
                with gr.TabItem("🎭 Scene"):
                    scene_tab = gr.Markdown()
                with gr.TabItem("🎯 Pitch"):
                    pitch_tab = gr.Markdown()
        
        generate_btn.click(
            fn=generate_story,
            inputs=[user_input],
            outputs=[story_output, plot_tab, char_tab, scene_tab, pitch_tab]
        )
        
        user_input.submit(
            fn=generate_story,
            inputs=[user_input],
            outputs=[story_output, plot_tab, char_tab, scene_tab, pitch_tab]
        )

# Launch
if __name__ == "__main__":
    demo.launch()