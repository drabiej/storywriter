#!/usr/bin/env python3
"""
Generate story outlines and character profiles from seed files.
"""

import os
import sys
import argparse
import anthropic
from pathlib import Path
from dotenv import load_dotenv
import datetime
import json
import time

# Load environment variables from .env file
load_dotenv()

def read_seed(seed_path: Path) -> str:
    """Read the seed file content."""
    with open(seed_path, 'r') as f:
        return f.read()

def generate_initial_outline(client: anthropic.Anthropic, seed_content: str) -> str:
    """Generate an initial outline from the seed content."""
    print("\n[1/8] Generating initial outline...", flush=True)
    print("  ⏳ Making API call to Claude...", flush=True)
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=20000,
        temperature=1.0,
        system="You are a skilled creative writer capable of developing detailed story outlines.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Create a detailed story outline based on the following seed:
                
                {seed_content}
                
                Structure your outline with:
                1. An engaging opening scene
                2. A clear narrative arc with rising action
                3. Character development opportunities
                4. 15-20 key scenes or chapters
                5. A satisfying conclusion
                
                IMPORTANT: Structure each chapter using the format "#### CHAPTER X: TITLE" (e.g., "#### CHAPTER 1: THE BEGINNING").
                This format is required for proper chapter detection by the story generation script.
                
                Focus on delivering a cohesive narrative that follows the themes, setting, and character elements from the seed.
                """
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 16000
        }
    )
    
    print("  ✅ Initial outline generated!", flush=True)
    
    try:
        # Extract the text content from the message properly handling different block types
        content_block = message.content[0]
        
        # If it's a ThinkingBlock, we need to get the text content differently
        if hasattr(content_block, 'thinking'):
            # For ThinkingBlock, get text from second content block which should be TextBlock
            if len(message.content) > 1 and hasattr(message.content[1], 'text'):
                content_text = message.content[1].text
            else:
                # Fallback if structure isn't as expected
                content_text = "Unable to extract content properly from thinking block response."
        else:
            # Regular TextBlock
            content_text = content_block.text
        
        # Print a small preview of the outline
        preview = content_text[:150] + "..."
        print(f"  📝 Preview: {preview}", flush=True)
        
        return content_text
    except Exception as e:
        print(f"  ⚠️ Error extracting content: {e}", flush=True)
        print(f"  ℹ️ Message structure: {message}", flush=True)
        print(f"  ℹ️ Content blocks: {[type(block) for block in message.content]}", flush=True)
        raise

def critique_outline(client: anthropic.Anthropic, outline: str, seed_content: str) -> str:
    """Critique the outline based on the seed requirements."""
    print("\n[2/8] Critiquing the outline...", flush=True)
    print("  ⏳ Making API call to Claude for critique...", flush=True)
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=20000,
        temperature=1.0,
        system="You are a discerning literary editor with expertise in critiquing story outlines.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Review this outline against the seed requirements.
                
                SEED:
                {seed_content}
                
                OUTLINE:
                {outline}
                
                Provide a detailed critique focusing on:
                1. How well the outline adheres to the seed's specified themes, characters, and setting
                2. Narrative coherence and pacing
                3. Character development opportunities
                4. Missing elements from the seed
                5. Specific suggestions for improvement
                """
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 16000
        }
    )
    
    print("  ✅ Critique completed!", flush=True)
    
    try:
        # Extract the text content from the message properly handling different block types
        content_block = message.content[0]
        
        # If it's a ThinkingBlock, we need to get the text content differently
        if hasattr(content_block, 'thinking'):
            # For ThinkingBlock, get text from second content block which should be TextBlock
            if len(message.content) > 1 and hasattr(message.content[1], 'text'):
                content_text = message.content[1].text
            else:
                # Fallback if structure isn't as expected
                content_text = "Unable to extract content properly from thinking block response."
        else:
            # Regular TextBlock
            content_text = content_block.text
        
        # Print a small preview of the critique
        preview = content_text[:150] + "..."
        print(f"  🔍 Preview: {preview}", flush=True)
        
        return content_text
    except Exception as e:
        print(f"  ⚠️ Error extracting critique content: {e}", flush=True)
        print(f"  ℹ️ Message structure: {message}", flush=True)
        print(f"  ℹ️ Content blocks: {[type(block) for block in message.content]}", flush=True)
        raise

def generate_character_profiles(client: anthropic.Anthropic, outline: str, seed_content: str) -> str:
    """Generate character profiles based on the outline and seed content."""
    print("\n[3/8] Generating initial character profiles...", flush=True)
    print("  ⏳ Making API call to Claude for character profiles...", flush=True)
    
    character_template = """
# Character Profile Template

## Basic Information
Name: 
Age:
Gender:
Physical Description:
Occupation:

## Background
Place of Birth:
Family History:
Education:
Significant Life Events:

## Personality
Core Traits:
Habits/Mannerisms:
Values:
Fears:
Desires:

## Relationships
Family:
Friends:
Enemies:
Romantic Interests:

## Character Development
Goals:
Motivations:
Conflicts:
Character Arc:

## Additional Details
Skills/Abilities:
Favorite Possessions:
Quirks:
Secrets:

## Story Role
Function in Plot:
Character Type (Protagonist, Antagonist, etc.):
Symbolic Meaning:
"""
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=20000,
        temperature=1.0,
        system="You are a skilled character developer capable of creating detailed and psychologically complex character profiles for fiction.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Based on the story outline and seed content provided, create detailed character profiles for all major characters in this story.
                
                SEED CONTENT:
                {seed_content}
                
                OUTLINE:
                {outline}
                
                For each major character (typically 3-6 characters), create a complete profile using the following template:
                
                {character_template}
                
                Identify the protagonist, any antagonists, and important supporting characters. 
                Create detailed, psychologically nuanced profiles that fit with the story world and narrative.
                Make sure each character has clear motivations, conflicts, and a definite arc within the story.
                Ensure consistency with any character details already established in the seed or outline.
                """
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 16000
        }
    )
    
    print("  ✅ Initial character profiles generated!", flush=True)
    
    try:
        # Extract the text content from the message properly handling different block types
        content_block = message.content[0]
        
        # If it's a ThinkingBlock, we need to get the text content differently
        if hasattr(content_block, 'thinking'):
            # For ThinkingBlock, get text from second content block which should be TextBlock
            if len(message.content) > 1 and hasattr(message.content[1], 'text'):
                content_text = message.content[1].text
            else:
                # Fallback if structure isn't as expected
                content_text = "Unable to extract content properly from thinking block response."
        else:
            # Regular TextBlock
            content_text = content_block.text
        
        # Print a small preview
        preview = content_text[:150] + "..."
        print(f"  👤 Preview: {preview}", flush=True)
        
        return content_text
    except Exception as e:
        print(f"  ⚠️ Error extracting character profiles content: {e}", flush=True)
        print(f"  ℹ️ Message structure: {message}", flush=True)
        print(f"  ℹ️ Content blocks: {[type(block) for block in message.content]}", flush=True)
        raise

def critique_character_profiles(client: anthropic.Anthropic, profiles: str, outline: str, seed_content: str) -> str:
    """Critique the character profiles for realism and consistency."""
    print("\n[4/8] Critiquing character profiles...", flush=True)
    print("  ⏳ Making API call to Claude for profile critique...", flush=True)
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=20000,
        temperature=1.0,
        system="You are a discerning character development expert with expertise in creating realistic, nuanced fictional characters.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Review these character profiles against the story outline and seed requirements.
                
                SEED:
                {seed_content}
                
                OUTLINE:
                {outline}
                
                CHARACTER PROFILES:
                {profiles}
                
                Provide a detailed critique focusing on:
                1. Character realism and psychological depth
                2. Internal consistency for each character
                3. Consistency with the story world and setting
                4. Character relationships and dynamics
                5. Character arcs and development potential
                6. Missed opportunities for conflict or tension
                7. Stereotypes or clichés to avoid
                
                For each character, offer specific suggestions for improvement that would make them more:
                - Psychologically complex and realistic
                - Distinctive and memorable
                - Integrated with the story's themes and setting
                """
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 16000
        }
    )
    
    print("  ✅ Character profile critique completed!", flush=True)
    
    try:
        # Extract the text content from the message properly handling different block types
        content_block = message.content[0]
        
        # If it's a ThinkingBlock, we need to get the text content differently
        if hasattr(content_block, 'thinking'):
            # For ThinkingBlock, get text from second content block which should be TextBlock
            if len(message.content) > 1 and hasattr(message.content[1], 'text'):
                content_text = message.content[1].text
            else:
                # Fallback if structure isn't as expected
                content_text = "Unable to extract content properly from thinking block response."
        else:
            # Regular TextBlock
            content_text = content_block.text
        
        # Print a small preview
        preview = content_text[:150] + "..."
        print(f"  🔍 Profile critique preview: {preview}", flush=True)
        
        return content_text
    except Exception as e:
        print(f"  ⚠️ Error extracting character critique content: {e}", flush=True)
        print(f"  ℹ️ Message structure: {message}", flush=True)
        print(f"  ℹ️ Content blocks: {[type(block) for block in message.content]}", flush=True)
        raise

def improve_character_profiles(client: anthropic.Anthropic, profiles: str, profile_critique: str, 
                               outline: str, seed_content: str) -> str:
    """Improve the character profiles based on the critique."""
    print("\n[5/8] Improving character profiles based on critique...", flush=True)
    print("  ⏳ Making API call to Claude for improved profiles...", flush=True)
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=20000,
        temperature=1.0,
        system="You are a skilled character developer capable of creating detailed, psychologically complex and realistic character profiles for fiction.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Revise the following character profiles based on the critique provided.
                
                ORIGINAL SEED:
                {seed_content}
                
                OUTLINE:
                {outline}
                
                ORIGINAL PROFILES:
                {profiles}
                
                CRITIQUE:
                {profile_critique}
                
                Create improved character profiles that:
                1. Address all critique points
                2. Enhance psychological realism and depth
                3. Remove any stereotypes or clichés
                4. Strengthen character relationships and dynamics
                5. Ensure each character has a clear and compelling arc
                6. Add complexity to motivations and conflicts
                7. Maintain consistency with the story world and setting
                
                Use the same format as the original profiles, but with enhanced content that addresses the critique.
                Focus on making each character more nuanced, realistic, and integrated with the story's themes.
                """
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 16000
        }
    )
    
    print("  ✅ Improved character profiles generated!", flush=True)
    
    try:
        # Extract the text content from the message properly handling different block types
        content_block = message.content[0]
        
        # If it's a ThinkingBlock, we need to get the text content differently
        if hasattr(content_block, 'thinking'):
            # For ThinkingBlock, get text from second content block which should be TextBlock
            if len(message.content) > 1 and hasattr(message.content[1], 'text'):
                content_text = message.content[1].text
            else:
                # Fallback if structure isn't as expected
                content_text = "Unable to extract content properly from thinking block response."
        else:
            # Regular TextBlock
            content_text = content_block.text
        
        # Print a small preview
        preview = content_text[:150] + "..."
        print(f"  👤 Improved profiles preview: {preview}", flush=True)
        
        return content_text
    except Exception as e:
        print(f"  ⚠️ Error extracting improved character profiles content: {e}", flush=True)
        print(f"  ℹ️ Message structure: {message}", flush=True)
        print(f"  ℹ️ Content blocks: {[type(block) for block in message.content]}", flush=True)
        raise

def generate_story_hooks(client: anthropic.Anthropic, outline: str, character_profiles: str, 
                          seed_content: str) -> str:
    """Generate strategic story hooks based on the outline and character profiles."""
    print("\n[6/8] Generating story hooks...", flush=True)
    print("  ⏳ Making API call to Claude for story hooks...", flush=True)
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=20000,
        temperature=1.0,
        system="You are a skilled narrative designer with expertise in creating engaging story hooks and tension points.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Analyze the provided story outline and character profiles and provide strategic recommendations 
                for where to place "hooks" and when to resolve them to maximize reader engagement.
                
                SEED CONTENT:
                {seed_content}
                
                OUTLINE:
                {outline}
                
                CHARACTER PROFILES:
                {character_profiles}
                
                ## Instructions for Hook Analysis
                
                When reviewing the story outline, please:
                
                1. First, read through the entire outline to understand the story arc, key characters, and main conflicts.
                
                2. Identify strategic positions to place hooks based on this structure:
                
                   ### At the Beginning (First 25%)
                   - Suggest a compelling opening hook for the first page/scene
                   - Identify where to place a hook at the inciting incident (around 10% mark)
                   - Recommend a significant hook at the first plot turn (25% mark)
                
                   ### In the Middle (Middle 50%)
                   - Highlight 2-3 places to add mid-story hooks that complicate the journey
                   - Suggest a major revelation hook at the midpoint (50% mark)
                   - Identify a "dark moment" hook opportunity near the 75% mark
                
                   ### Near the End (Final 25%)
                   - Recommend a final tension hook before the climax
                   - Suggest any epilogue hooks if appropriate
                
                3. For each hook recommendation, please:
                   - Explain specifically what type of hook would work best at this point
                   - Provide a concrete suggestion tied to the story elements
                   - Explain why this hook will be effective at this position
                
                4. Create a hook resolution schedule by:
                   - Recommending when each hook should be resolved
                   - Identifying which hooks should receive partial revelations
                   - Suggesting how resolving one hook can introduce a new, larger question
                
                5. Evaluate the existing outline for:
                   - Sections where reader interest might wane
                   - Areas where too many hooks are resolved at once
                   - Opportunities to connect character arcs with plot hooks
                
                ## Example Format for Your Analysis
                
                For each section of the outline, please structure your recommendations like this:
                
                ```
                SECTION: [Chapter/Scene]
                
                CURRENT TENSION: [Assessment of current reader engagement]
                
                HOOK RECOMMENDATION: 
                - Type: [Question/Mystery/Promise/Contradiction]
                - Suggestion: [Specific hook idea]
                - Placement: [Exactly where in the scene]
                - Reader Effect: [What this makes readers wonder]
                
                RESOLUTION STRATEGY:
                - When to resolve: [Chapter/percentage point]
                - Method: [Full/Partial revelation]
                - Next hook: [What new question this resolution raises]
                ```
                
                Please be specific to the story elements rather than giving generic advice, and focus on creating 
                a rhythm of tension and release that maintains momentum throughout the narrative.
                """
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 16000
        }
    )
    
    print("  ✅ Story hooks generated!", flush=True)
    
    try:
        # Extract the text content from the message properly handling different block types
        content_block = message.content[0]
        
        # If it's a ThinkingBlock, we need to get the text content differently
        if hasattr(content_block, 'thinking'):
            # For ThinkingBlock, get text from second content block which should be TextBlock
            if len(message.content) > 1 and hasattr(message.content[1], 'text'):
                content_text = message.content[1].text
            else:
                # Fallback if structure isn't as expected
                content_text = "Unable to extract content properly from thinking block response."
        else:
            # Regular TextBlock
            content_text = content_block.text
        
        # Print a small preview
        preview = content_text[:150] + "..."
        print(f"  🎣 Story hooks preview: {preview}", flush=True)
        
        return content_text
    except Exception as e:
        print(f"  ⚠️ Error extracting story hooks content: {e}", flush=True)
        print(f"  ℹ️ Message structure: {message}", flush=True)
        print(f"  ℹ️ Content blocks: {[type(block) for block in message.content]}", flush=True)
        raise

def save_story_hooks(hooks: str, seed_path: Path) -> Path:
    """Save story hooks to a file."""
    print("\n[✍️] Saving story hooks...", flush=True)
    
    # Create hooks directory if it doesn't exist
    hooks_dir = Path('hooks')
    hooks_dir.mkdir(exist_ok=True)
    
    seed_name = seed_path.stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hooks_filename = f"{seed_name}_hooks_{timestamp}.md"
    hooks_path = hooks_dir / hooks_filename
    
    with open(hooks_path, 'w') as f:
        f.write(hooks)
    
    print(f"  💾 Story hooks saved as: {hooks_path}", flush=True)
    
    return hooks_path

def improve_outline_with_profiles_and_hooks(client: anthropic.Anthropic, outline: str, critique: str, 
                                          character_profiles: str, story_hooks: str, seed_content: str) -> str:
    """Improve the outline based on the critique, character profiles, and story hooks."""
    print("\n[7/8] Improving outline with character profiles and story hooks...", flush=True)
    print("  ⏳ Making API call to Claude for improved outline...", flush=True)
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=20000,
        temperature=1.0,
        system="You are a skilled creative writer capable of revising story outlines based on critical feedback, character profiles, and narrative hooks.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Revise the following story outline based on the critique provided, the detailed character profiles, and the story hooks analysis.
                
                ORIGINAL SEED:
                {seed_content}
                
                ORIGINAL OUTLINE:
                {outline}
                
                CRITIQUE:
                {critique}
                
                CHARACTER PROFILES:
                {character_profiles}
                
                STORY HOOKS ANALYSIS:
                {story_hooks}
                
                Create an improved outline that:
                1. Addresses all the critique points
                2. Fully integrates the detailed character profiles
                3. Implements the recommended story hooks at strategic points
                4. Follows the hook resolution schedule to maintain tension
                5. Maintains the core elements from the original seed
                6. Ensures character motivations and arcs are clearly reflected in the plot
                7. Takes advantage of character relationships and conflicts
                8. Creates opportunities for character growth and development
                
                IMPORTANT: Structure the chapters using the format "#### CHAPTER X: TITLE" (e.g., "#### CHAPTER 1: THE BEGINNING").
                This format is required for proper chapter detection by the story generation script.
                
                The final outline should be comprehensive, well-structured, and ready for story development,
                with character actions and decisions flowing naturally from their established profiles.
                Strategic hooks should be clearly integrated to create ongoing reader engagement.
                """
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 16000
        }
    )
    
    print("  ✅ Improved outline with character profiles and hooks generated!", flush=True)
    
    try:
        # Extract the text content from the message properly handling different block types
        content_block = message.content[0]
        
        # If it's a ThinkingBlock, we need to get the text content differently
        if hasattr(content_block, 'thinking'):
            # For ThinkingBlock, get text from second content block which should be TextBlock
            if len(message.content) > 1 and hasattr(message.content[1], 'text'):
                content_text = message.content[1].text
            else:
                # Fallback if structure isn't as expected
                content_text = "Unable to extract content properly from thinking block response."
        else:
            # Regular TextBlock
            content_text = content_block.text
        
        # Print a small preview of the final outline
        preview = content_text[:150] + "..."
        print(f"  📝 Preview: {preview}", flush=True)
        
        return content_text
    except Exception as e:
        print(f"  ⚠️ Error extracting improved outline content: {e}", flush=True)
        print(f"  ℹ️ Message structure: {message}", flush=True)
        print(f"  ℹ️ Content blocks: {[type(block) for block in message.content]}", flush=True)
        raise

def save_outline(final_outline: str, seed_path: Path) -> Path:
    """Save the final outline to a file."""
    print("\n[8/8] Saving final outline...", flush=True)
    
    outlines_dir = Path('outlines')
    outlines_dir.mkdir(exist_ok=True)
    
    seed_name = seed_path.stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outline_filename = f"{seed_name}_{timestamp}.md"
    outline_path = outlines_dir / outline_filename
    
    with open(outline_path, 'w') as f:
        f.write(final_outline)
    
    print(f"  💾 Outline saved as: {outline_path}", flush=True)
    
    return outline_path

def display_progress_update():
    """Display a real-time progress update."""
    print("\n[📊] Story development in progress...", flush=True)
    for i in range(10):
        sys.stdout.write("\r  ⏳ Processing... " + "▓" * i + "░" * (10-i) + f" {i*10}%")
        sys.stdout.flush()
        time.sleep(0.5)
    print("\r  ⏳ Processing... ▓▓▓▓▓▓▓▓▓▓ 100%")
    print("  ✨ Next phase starting soon...", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Generate story outlines and character profiles from seed files")
    parser.add_argument("--seed", type=str, required=True, help="Path to the seed file")
    parser.add_argument("--skip-profiles", action="store_true", help="Skip character profile generation")
    parser.add_argument("--skip-profile-critique", action="store_true", help="Skip the character profile critique and improvement")
    parser.add_argument("--skip-hooks", action="store_true", help="Skip the story hooks generation")
    args = parser.parse_args()
    
    seed_path = Path(args.seed)
    if not seed_path.exists():
        print(f"❌ Error: Seed file '{seed_path}' not found.")
        sys.exit(1)
    
    try:
        print(f"🔄 Starting story development process for seed: {seed_path.name}", flush=True)
        print(f"⏱️  This process may take several minutes to complete...", flush=True)
        
        start_time = datetime.datetime.now()
        
        client = anthropic.Anthropic()
        
        # Read seed content
        seed_content = read_seed(seed_path)
        print(f"📖 Seed file loaded: {len(seed_content)} characters", flush=True)
        
        # Generate initial outline
        initial_outline = generate_initial_outline(client, seed_content)
        display_progress_update()
        
        # Critique the outline
        critique = critique_outline(client, initial_outline, seed_content)
        display_progress_update()
        
        # Character profile generation and improvement
        if args.skip_profiles:
            print("\n[3/8 - 5/8] Skipping character profile generation as requested...", flush=True)
            final_character_profiles = ""
            profiles_path = None
        else:
            # Generate initial character profiles
            initial_profiles = generate_character_profiles(client, initial_outline, seed_content)
            display_progress_update()
            
            if args.skip_profile_critique:
                print("\n[4/8 & 5/8] Skipping character profile critique and improvement as requested...", flush=True)
                final_character_profiles = initial_profiles
            else:
                # Critique character profiles
                profile_critique = critique_character_profiles(client, initial_profiles, initial_outline, seed_content)
                display_progress_update()
                
                # Improve character profiles based on critique
                final_character_profiles = improve_character_profiles(
                    client, initial_profiles, profile_critique, initial_outline, seed_content
                )
                display_progress_update()
            
            # Save final character profiles
            # Add the missing save_character_profiles function
            def save_character_profiles(profiles: str, seed_path: Path) -> Path:
                """Save character profiles to a file."""
                print("\n[✍️] Saving character profiles...", flush=True)
                
                # Create profiles directory if it doesn't exist
                profiles_dir = Path('profiles')
                profiles_dir.mkdir(exist_ok=True)
                
                seed_name = seed_path.stem
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                profiles_filename = f"{seed_name}_profiles_{timestamp}.md"
                profiles_path = profiles_dir / profiles_filename
                
                with open(profiles_path, 'w') as f:
                    f.write(profiles)
                
                print(f"  💾 Character profiles saved as: {profiles_path}", flush=True)
                
                return profiles_path
                
            profiles_path = save_character_profiles(final_character_profiles, seed_path)
        
        # Generate story hooks
        if args.skip_hooks:
            print("\n[6/8] Skipping story hooks generation as requested...", flush=True)
            story_hooks = ""
            hooks_path = None
        else:
            story_hooks = generate_story_hooks(client, initial_outline, final_character_profiles, seed_content)
            hooks_path = save_story_hooks(story_hooks, seed_path)
            display_progress_update()
        
        # Improve the outline based on critique, character profiles, and story hooks
        final_outline = improve_outline_with_profiles_and_hooks(
            client, initial_outline, critique, final_character_profiles, story_hooks, seed_content
        )
        display_progress_update()
        
        # Save the final outline
        outline_path = save_outline(final_outline, seed_path)
        
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        minutes, seconds = divmod(total_time.seconds, 60)
        
        print(f"\n✨ Process completed in {minutes} minutes and {seconds} seconds", flush=True)
        print(f"📚 Final outline saved to: {outline_path}", flush=True)
        if profiles_path:
            print(f"👤 Character profiles saved to: {profiles_path}", flush=True)
        if hooks_path:
            print(f"🎣 Story hooks saved to: {hooks_path}", flush=True)
        print(f"🎉 Done! You can now view your story materials in the respective directories.", flush=True)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
