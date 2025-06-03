# app.py - Fixed structure for Hugging Face Spaces
import gradio as gr
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Get API key from Hugging Face Secrets
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    # Create interface for missing API key
    def no_api_key_interface():
        return """
        ⚠️ **OpenAI API Key Required**
        
        To use this app, you need to add an OpenAI API key:
        
        **For Space Owners:**
        1. Go to your Space Settings
        2. Click on "Repository secrets"  
        3. Add a new secret named: `OPENAI_API_KEY`
        4. Paste your OpenAI API key as the value
        5. Restart the space
        
        **Get an API key:** [OpenAI API Keys](https://platform.openai.com/api-keys)
        """
    
    demo = gr.Interface(
        fn=lambda x: no_api_key_interface(),
        inputs=gr.Textbox(label="Story Idea", placeholder="API key required"),
        outputs=gr.Markdown(),
        title="🎬 AI Story Generator - Setup Required"
    )

else:
    # Initialize LLM (inside the else block)
    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0.7,
        max_tokens=2500,
        openai_api_key=api_key,
        request_timeout=90
    )

    def create_story_generator():
        """Create the story generation pipeline"""
        output_parser = StrOutputParser()
        
        # Story Concept Prompt
        concept_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""Based on this user input: {user_input}

Generate a detailed story concept (aim for 200-250 words) that includes:

CHARACTER: Create a compelling main character with name, age, background, and unique traits
SETTING: Establish a vivid time and place with specific details
CONFLICT: Define a central challenge or problem that drives the story
GENRE: Identify the story type and any subgenres

Format your response clearly:

CHARACTER: [Name, age, detailed description including personality, background, and what makes them unique]

SETTING: [Specific time period and location with atmospheric details]

CONFLICT: [The main problem or challenge the character faces, including stakes and obstacles]

GENRE: [Primary genre and any subgenres, with brief explanation of story tone]

Write the complete concept:"""
        )
        
        # Plot Outline Prompt
        plot_prompt = PromptTemplate(
            input_variables=["story_concept"],
            template="""Based on this story concept:
{story_concept}

Create a comprehensive 5-act plot outline (aim for 400-500 words total):

ACT 1 - SETUP (Opening):
Write 3-4 sentences establishing the world, introducing the protagonist, and hinting at the coming conflict.

ACT 2 - RISING ACTION (Development):
Write 3-4 sentences showing how the conflict emerges, stakes escalate, and complications arise.

ACT 3 - CLIMAX (Peak Tension):
Write 3-4 sentences describing the highest point of tension, major confrontation, and crucial turning point.

ACT 4 - FALLING ACTION (Consequences):
Write 3-4 sentences showing the aftermath of the climax, character growth, and how resolution begins.

ACT 5 - RESOLUTION (Ending):
Write 3-4 sentences explaining how conflicts are resolved, character arcs complete, and the new status quo.

Write the complete 5-act structure:"""
        )
        
        # Character Development Prompt
        character_prompt = PromptTemplate(
            input_variables=["story_concept", "plot_outline"],
            template="""Based on these story elements:

STORY CONCEPT:
{story_concept}

PLOT OUTLINE:
{plot_outline}

Develop 4 detailed characters (aim for 350-400 words total):

PROTAGONIST:
Name and Age: [Full name and specific age]
Background: [Personal history, family, education, key life events]
Personality: [3-4 key personality traits with examples]
Motivations: [What drives them, their goals and fears]
Character Arc: [How they change throughout the story]

ANTAGONIST:
Name and Age: [Full name and specific age]
Background: [Personal history and what shaped them]
Personality: [3-4 key traits, including flaws and strengths]
Motivations: [Why they oppose the protagonist, their goals]
Methods: [How they create conflict, their approach]

SUPPORTING CHARACTER #1:
Name and Age: [Full name and specific age]
Role: [Their function in the story]
Personality: [2-3 key traits]
Relationship: [Connection to protagonist and their dynamic]

SUPPORTING CHARACTER #2:
Name and Age: [Full name and specific age]
Role: [Their function in the story]
Personality: [2-3 key traits]
Relationship: [Connection to other characters and story impact]

Write all character profiles completely:"""
        )
        
        # Opening Scene Prompt
        scene_prompt = PromptTemplate(
            input_variables=["story_concept", "plot_outline", "characters"],
            template="""Using all these story elements:

CONCEPT: {story_concept}
PLOT: {plot_outline}
CHARACTERS: {characters}

Write a compelling opening scene (aim for 300-400 words) that accomplishes:

SETTING ESTABLISHMENT: Use sensory details (sight, sound, smell, touch, taste) to make the world feel real
CHARACTER INTRODUCTION: Show the protagonist in action, revealing personality through behavior
CONFLICT FORESHADOWING: Plant subtle hints about the central conflict without giving everything away
TONE AND MOOD: Establish the story's atmosphere and emotional feeling
HOOK: Create intrigue that makes readers want to continue

Write the complete opening scene with vivid, immersive description:"""
        )
        
        # Marketing Pitch Prompt
        pitch_prompt = PromptTemplate(
            input_variables=["story_concept", "plot_outline", "characters", "opening_scene"],
            template="""Based on the complete story development:

CONCEPT: {story_concept}
PLOT: {plot_outline}
CHARACTERS: {characters}
OPENING SCENE: {opening_scene}

Create a professional marketing pitch (aim for 300-350 words) that includes:

TITLE:
[Create a compelling, memorable title that captures the story's essence]

LOGLINE:
[Write one powerful sentence that encapsulates the entire story and hooks interest]

SYNOPSIS:
[Provide a 4-5 sentence summary highlighting the unique elements, main conflict, and what makes this story special]

TARGET AUDIENCE:
[Identify the specific demographic who would love this story, including age range, interests, and reading preferences]

COMPARABLE WORKS:
[List 2-3 successful books, movies, or shows with similar themes/style and explain why fans of those would enjoy this story]

MARKET APPEAL:
[Explain why this story would succeed commercially, what trends it taps into, and its unique selling points]

Write the complete professional pitch:"""
        )
        
        # Create the individual chains
        concept_chain = concept_prompt | llm | output_parser
        plot_chain = plot_prompt | llm | output_parser
        character_chain = character_prompt | llm | output_parser
        scene_chain = scene_prompt | llm | output_parser
        pitch_chain = pitch_prompt | llm | output_parser
        
        def generate_complete_story(inputs):
            """Run the complete story generation pipeline"""
            try:
                user_input = inputs["user_input"]
                print(f"🎬 Starting story generation for: '{user_input[:50]}...'")
                
                # Step 1: Generate Story Concept
                print("📝 Step 1/5: Creating story concept...")
                concept_result = concept_chain.invoke({"user_input": user_input})
                print(f"   ✅ Concept generated ({len(concept_result)} characters)")
                
                # Step 2: Generate Plot Outline
                print("📋 Step 2/5: Developing plot outline...")
                plot_result = plot_chain.invoke({"story_concept": concept_result})
                print(f"   ✅ Plot generated ({len(plot_result)} characters)")
                
                # Step 3: Develop Characters
                print("👥 Step 3/5: Creating character profiles...")
                character_result = character_chain.invoke({
                    "story_concept": concept_result,
                    "plot_outline": plot_result
                })
                print(f"   ✅ Characters generated ({len(character_result)} characters)")
                
                # Step 4: Write Opening Scene
                print("🎭 Step 4/5: Writing opening scene...")
                scene_result = scene_chain.invoke({
                    "story_concept": concept_result,
                    "plot_outline": plot_result,
                    "characters": character_result
                })
                print(f"   ✅ Scene generated ({len(scene_result)} characters)")
                
                # Step 5: Create Marketing Pitch
                print("🎯 Step 5/5: Creating marketing pitch...")
                pitch_result = pitch_chain.invoke({
                    "story_concept": concept_result,
                    "plot_outline": plot_result,
                    "characters": character_result,
                    "opening_scene": scene_result
                })
                print(f"   ✅ Pitch generated ({len(pitch_result)} characters)")
                
                print("🎉 Story generation completed successfully!")
                
                # Calculate total output length
                total_length = sum([
                    len(concept_result), len(plot_result), len(character_result),
                    len(scene_result), len(pitch_result)
                ])
                print(f"📊 Total output: {total_length} characters")
                
                return {
                    "story_concept": concept_result,
                    "plot_outline": plot_result,
                    "characters": character_result,
                    "opening_scene": scene_result,
                    "marketing_pitch": pitch_result,
                    "total_length": total_length
                }
                
            except Exception as e:
                print(f"❌ Error during story generation: {str(e)}")
                raise Exception(f"Story generation failed: {str(e)}")
        
        return RunnableLambda(generate_complete_story)

    # Initialize the story generator (inside the else block)
    story_generator = create_story_generator()

    def generate_story(user_input, progress=gr.Progress()):
        """Main function called by Gradio interface"""
        
        # Input validation
        if not user_input or not user_input.strip():
            return "💡 **Please enter a story idea!**\n\nTry: 'A detective who can see object histories'", "", "", "", ""
        
        if len(user_input.strip()) < 10:
            return "📝 **Please provide more detail** (at least 10 characters)", "", "", "", ""
        
        if len(user_input.strip()) > 500:
            return "✂️ **Story idea too long** (max 500 characters)", "", "", "", ""
        
        try:
            progress(0.1, desc="🚀 Starting story generation...")
            
            # Generate the story
            result = story_generator.invoke({"user_input": user_input.strip()})
            
            progress(0.9, desc="✨ Formatting results...")
            
            # Format the outputs
            concept = f"""## 📝 Story Concept
*Generated: {len(result['story_concept'])} characters*

{result['story_concept']}

---"""

            plot = f"""## 📋 Complete Plot Outline
*Generated: {len(result['plot_outline'])} characters*

{result['plot_outline']}

---"""

            characters = f"""## 👥 Character Profiles
*Generated: {len(result['characters'])} characters*

{result['characters']}

---"""

            scene = f"""## 🎭 Opening Scene
*Generated: {len(result['opening_scene'])} characters*

{result['opening_scene']}

---"""

            pitch = f"""## 🎯 Professional Marketing Pitch
*Generated: {len(result['marketing_pitch'])} characters*

{result['marketing_pitch']}

---
**Total Story Package:** {result['total_length']} characters | **Model:** GPT-3.5-Turbo-Instruct"""
            
            progress(1.0, desc="🎉 Complete!")
            return concept, plot, characters, scene, pitch
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "rate limit" in error_str:
                error_msg = "⏱️ **Rate limit reached.** Please wait a few minutes and try again."
            elif "api key" in error_str:
                error_msg = "🔑 **API key issue.** Please check the space configuration."
            elif "timeout" in error_str:
                error_msg = "⏰ **Request timed out.** Try a shorter idea or try again."
            else:
                error_msg = f"❌ **Error:** {str(e)}\n\nPlease try again with a different idea."
            
            return error_msg, "", "", "", ""

    # Create the Gradio interface (inside the else block)
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="🎬 AI Story Generator"
    ) as demo:
        
        gr.Markdown("""
        # 🎬 AI Story Generator
        ### Transform your ideas into complete story concepts
        
        **Powered by:** OpenAI GPT-3.5-Turbo-Instruct
        
        Enter a story idea and get: concept, plot, characters, opening scene, and marketing pitch!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                user_input = gr.Textbox(
                    label="💡 Your Story Idea",
                    placeholder="A teenager discovers they can communicate with plants...",
                    lines=5,
                    max_lines=8
                )
                
                generate_btn = gr.Button(
                    "✨ Generate Complete Story", 
                    variant="primary", 
                    size="lg"
                )
                
                gr.Markdown("""
                ### 💫 Example Ideas:
                - A detective who can see object histories
                - Time stops except for librarians  
                - Plants gossip through social media
                - A memory thief's forgotten past
                - Lies become temporarily true
                """)
        
        with gr.Column(scale=2):
            story_output = gr.Markdown("Your story package will appear here...")
            
            with gr.Tabs():
                with gr.TabItem("📋 Plot"):
                    plot_tab = gr.Markdown()
                with gr.TabItem("👥 Characters"):
                    char_tab = gr.Markdown()
                with gr.TabItem("🎭 Scene"):
                    scene_tab = gr.Markdown()
                with gr.TabItem("🎯 Pitch"):
                    pitch_tab = gr.Markdown()
        
        # Event handlers
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

# Launch the app
if __name__ == "__main__":
    demo.launch()