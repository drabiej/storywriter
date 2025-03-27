#!/usr/bin/env python3
"""
Generate a story chapter by chapter using Gemini API.

This script starts from a seed file (premise) and generates a complete story
chapter by chapter without requiring a separate outline step. It creates a timestamped 
directory in the stories/<story_name>/ directory, with each chapter saved as a separate 
file and all prompts saved in stories/<story_name>/prompts/ for reference.

The chapter generation process has three steps:
1. Initial chapter generation - Create a first draft
2. Consistency revision - Fix plot continuity and character consistency
3. Style revision - Improve prose quality and readability

The script maintains story continuity by including previous chapters in the context
up to the specified maximum context size (default: 500,000 characters).

Usage:
    # Generate a story from a seed file (premise):
    ./generate_gemini.py --seed seeds/chalion.md --num-chapters 15
    ./generate_gemini.py --seed seeds/chalion.md --num-chapters 10 --genre fantasy
    ./generate_gemini.py --seed seeds/chalion.md --num-chapters 20 --skip-revisions
    ./generate_gemini.py --seed seeds/chalion.md --num-chapters 15 --revision-context 500000
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Callable, TypeVar, Any
from datetime import datetime
import base64
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Type variable for generic retry function
T = TypeVar('T')

def retry_with_exponential_backoff(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Retry a function with exponential backoff when read operations time out.
    
    Args:
        func: Function to retry
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
        
    Raises:
        Exception: If all retries fail
    """
    max_retries = 2  # Initial try + 2 retries
    retry_count = 0
    last_exception = None
    
    # Backoff delays: initial try, then 10 minutes, then 1 hour
    backoff_delays = [0, 600, 3600]  # in seconds
    
    while retry_count <= max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            # Check if it's a timeout error
            if "read operation timed out" in str(e).lower() or "timeout" in str(e).lower():
                retry_count += 1
                if retry_count <= max_retries:
                    delay = backoff_delays[retry_count]
                    print(f"\nRead operation timed out. Retrying in {delay} seconds (retry {retry_count}/{max_retries})...")
                    time.sleep(delay)
                else:
                    print(f"\nFailed after {max_retries} retries: {e}")
                    raise
            else:
                # Not a timeout error, don't retry
                raise
    
    # This should not be reached, but just in case
    raise last_exception

def extract_message_content(response) -> str:
    """Extract text content from a Gemini response message."""
    # For Gemini, response is directly the text
    return response

def setup_directories(story_dir: Path):
    """Ensure required directories exist."""
    # Create main story directory
    story_dir.mkdir(parents=True, exist_ok=True)
    
    # Create chapters directory
    chapters_dir = story_dir / "chapters"
    chapters_dir.mkdir(exist_ok=True)
    
    # Create summaries directory
    summaries_dir = chapters_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    
    # Create prompts directory
    prompts_dir = story_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    
    # Create templates directory
    templates_dir = prompts_dir / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Create responses directory
    responses_dir = story_dir / "responses"
    responses_dir.mkdir(exist_ok=True)
    
    # Ensure the global templates directory exists
    global_templates_dir = Path("simple_prompts/templates")
    global_templates_dir.mkdir(parents=True, exist_ok=True)

def extract_chapter_outline(seed_content: str, chapter_num: int) -> str:
    """Extract a detailed chapter outline from the seed content."""
    print(f"Extracting outline for Chapter {chapter_num} from seed content...")
    
    # Load the template for chapter outline extraction
    template_path = Path("simple_prompts/templates/extract_chapter_outline.txt")
    
    # Check if template exists, error out if not
    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found.")
        sys.exit(1)
    
    # Load the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders
    prompt = template.replace("{{SEED_CONTENT}}", seed_content)
    prompt = prompt.replace("{{CHAPTER_NUMBER}}", str(chapter_num))
    
    try:
        # Define a function to call Gemini and process the response
        def call_gemini_for_outline():
            # Configure the API key at the client level
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = GenerativeModel(
                model_name="gemini-2.5-pro-exp-03-25"
            )
            
            response_text = ""
            for chunk in model.generate_content(prompt, stream=True):
                response_text += chunk.text
                # Print a progress indicator
                print(".", end="", flush=True)
            
            print()  # New line after progress dots
            
            return response_text
        
        # Call with retry logic
        response = retry_with_exponential_backoff(call_gemini_for_outline)
        
        # Extract the chapter outline
        chapter_outline = extract_message_content(response)
        return chapter_outline
        
    except Exception as e:
        print(f"Error extracting chapter outline: {e}")
        return f"Error: Could not extract chapter outline due to an API error: {e}"

def generate_chapter(
    chapter_num: int,
    story_title: str,
    previous_chapters: List[str] = None,
    seed_content: str = None,
    temperature: float = 1.0,
    skip_revisions: bool = False,
    revision_context: int = 400000,
    writing_style: str = None,
    genre: str = "fiction"
) -> str:
    """Generate a single chapter using Gemini with context from previous chapters.
    
    This function performs a three-step process:
    1. Initial chapter generation
    2. Consistency/plot revision (checking for narrative coherence with prior chapters)
    3. Writing style revision (improving prose quality and conciseness)
    """
    
    print(f"Generating initial draft for Chapter {chapter_num}...")
    
    # Extract specific guidance for this chapter from the seed content
    if seed_content:
        chapter_outline = extract_chapter_outline(seed_content, chapter_num)
        print(f"Generated chapter outline for Chapter {chapter_num}")
    else:
        chapter_outline = f"Chapter {chapter_num} of the story."
    
    # Generic chapter title
    chapter_title = f"Chapter {chapter_num}"
    
    # Prepare previous chapters context
    previous_context = ""
    if previous_chapters:
        previous_context = "Previous chapters:\n\n"
        for i, chapter_text in enumerate(previous_chapters):
            # Use the most recent chapter numbers
            chapter_number = chapter_num - len(previous_chapters) + i
            if chapter_number > 0:
                previous_context += f"<chapter_{chapter_number}>\n{chapter_text}\n</chapter_{chapter_number}>\n\n"
    
    # Load the template for initial chapter generation
    template_path = Path("simple_prompts/templates/initial_chapter.txt")
    
    # Check if template exists, error out if not
    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found.")
        sys.exit(1)
    
    # Load the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Prepare style guidance if available
    style_guidance = ""
    if writing_style:
        style_guidance = f"""
WRITING STYLE GUIDANCE:
{writing_style}

Follow this writing style guidance carefully when crafting the chapter.
"""

    # Replace placeholders in template
    prompt = template.replace("{{CHAPTER_NUMBER}}", str(chapter_num))
    prompt = prompt.replace("{{CHAPTER_TITLE}}", chapter_title)
    prompt = prompt.replace("{{STORY_TITLE}}", story_title)
    prompt = prompt.replace("{{CHAPTER_DESCRIPTION}}", chapter_outline)
    prompt = prompt.replace("{{PREVIOUS_CONTEXT}}", previous_context)
    prompt = prompt.replace("{{STYLE_GUIDANCE}}", style_guidance)
    prompt = prompt.replace("{{SEED_CONTENT}}", seed_content if seed_content else "")
    prompt = prompt.replace("{{GENRE}}", genre)

    # Save the initial generation prompt
    if story_title:
        story_name = story_title.lower().replace(" ", "_")
    else:
        story_name = "unnamed_story"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get parent directory from context
    prompts_dir = Path("stories") / story_name / "prompts"
    os.makedirs(prompts_dir, exist_ok=True)
    
    prompt_path = prompts_dir / f"{story_name}_ch{chapter_num:02d}_generation_{timestamp}.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"  Saved initial generation prompt to {prompt_path}")
    
    try:
        start_time = time.time()
        
        # Define a function to call Gemini and process the response
        def call_gemini_for_generation():
            # Configure the API key at the client level
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = GenerativeModel(
                model_name="gemini-2.5-pro-exp-03-25"
            )
            
            response_text = ""
            for chunk in model.generate_content(prompt, stream=True):
                response_text += chunk.text
                # Print a progress indicator
                print(".", end="", flush=True)
            
            print()  # New line after progress dots
            
            return response_text
        
        # Call with retry logic
        response = retry_with_exponential_backoff(call_gemini_for_generation)
        
        # Calculate and display generation time
        elapsed_time = time.time() - start_time
        print(f"Initial draft generated in {elapsed_time:.1f} seconds")
        
        # Extract the chapter text
        draft_chapter = extract_message_content(response)
        
        # Save Gemini's response to the responses directory
        if story_title:
            story_name = story_title.lower().replace(" ", "_")
        else:
            story_name = "unnamed_story"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        responses_dir = Path("stories") / story_name / "responses"
        os.makedirs(responses_dir, exist_ok=True)
        response_path = responses_dir / f"{story_name}_ch{chapter_num:02d}_generation_{timestamp}.txt"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(draft_chapter)
        print(f"  Saved generation response to {response_path}")
        
        # If revisions are skipped, return the draft
        if skip_revisions:
            return draft_chapter
            
        # Step 2: Revise for consistency and plot coherence
        revised_chapter = revise_for_consistency(
            draft_chapter=draft_chapter,
            chapter_num=chapter_num,
            chapter_title=chapter_title,
            chapter_description=chapter_outline,
            story_title=story_title,
            previous_chapters=previous_chapters,
            max_context_chars=revision_context,  # Use parameter passed to this function
            writing_style=writing_style,
            seed_content=seed_content,
            genre=genre
        )
        
        # Step 3: Revise for writing style and quality
        final_chapter = revise_for_style(
            chapter_content=revised_chapter,
            chapter_num=chapter_num,
            chapter_title=chapter_title,
            story_title=story_title,
            writing_style=writing_style,
            seed_content=seed_content,
            genre=genre
        )
        
        return final_chapter
            
    except Exception as e:
        print(f"Error generating chapter: {e}")
        return f"Error generating Chapter {chapter_num}: {str(e)}"


def critique_for_consistency(
    draft_chapter: str,
    chapter_num: int,
    chapter_title: str,
    chapter_description: str,
    story_title: str,
    previous_chapters: List[str] = None,
    max_context_chars: int = 400000,  # Allow a large context for consistency checking
    writing_style: str = None,
    seed_content: str = None,
    genre: str = "fiction"
) -> str:
    """Critique chapter for narrative consistency and plot coherence issues.
    
    This function provides as many previous chapters as possible to ensure maximum consistency
    with the established narrative. Context management ensures we don't exceed Gemini's limits.
    """
    
    print(f"Critiquing Chapter {chapter_num} for narrative consistency and plot coherence...")
    
    # Prepare previous chapters context with context management
    previous_context = ""
    current_context_size = 0
    
    if previous_chapters:
        previous_context = "Previous chapters:\n\n"
        # Create a copy to work with
        chapters_for_context = previous_chapters.copy()
        selected_chapters = []
        
        # First add the most recent chapter, as it's most important for continuity
        if chapters_for_context:
            most_recent = chapters_for_context.pop()
            selected_chapters.append((chapter_num-1, most_recent))
            current_context_size += len(most_recent)
        
        # Now add as many LATER chapters as possible, starting from the most recent
        # Reverse the remaining chapters to prioritize later ones
        remaining_chapters = list(enumerate(chapters_for_context))
        remaining_chapters.reverse()  # Reverse to start with latest chapters
        
        for i, chapter_text in remaining_chapters:
            chapter_number = chapter_num - len(previous_chapters) + i
            if chapter_number <= 0:
                continue  # Skip invalid chapter numbers
                
            # Check if adding this chapter would exceed our context limit
            # Leave room for the draft chapter and prompt text (~50K chars)
            if current_context_size + len(chapter_text) + len(draft_chapter) + 50000 > max_context_chars:
                print(f"  Skipping chapter {chapter_number} to stay within context limit")
                continue
                
            selected_chapters.append((chapter_number, chapter_text))
            current_context_size += len(chapter_text)
            
        # Sort by chapter number so they're in proper order
        selected_chapters.sort(key=lambda x: x[0])
        
        # Add selected chapters to context
        for chapter_number, chapter_text in selected_chapters:
            previous_context += f"<chapter_{chapter_number}>\n{chapter_text}\n</chapter_{chapter_number}>\n\n"
            
        print(f"  Using {len(selected_chapters)} previous chapters in critique context ({current_context_size} chars)")
    else:
        previous_context = "No previous chapters available."
    
    # Add writing style to the critique template if available
    style_section = ""
    if writing_style:
        style_section = """
4. WRITING STYLE ADHERENCE
   - Is the chapter following the specified writing style guidance?
   - Are there areas where the style could be better maintained?
"""

    # Define story name from story title
    if story_title:
        story_name = story_title.lower().replace(" ", "_")
    else:
        story_name = "unnamed_story"
        
    # Define source and destination template paths
    source_template_path = Path("simple_prompts/templates/critique_consistency.txt")
    template_path = Path("stories") / story_name / "prompts" / "templates" / "critique_consistency.txt"
    
    # Create templates directory under the new structure
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    
    # Copy the template from source
    if os.path.exists(source_template_path):
        import shutil
        shutil.copy2(source_template_path, template_path)
    else:
        print(f"Error: Source template {source_template_path} not found.")
        sys.exit(1)
    
    # Load the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Prepare style_section and writing_style_section 
    style_section = ""
    writing_style_section = ""
    
    if writing_style:
        style_section = """
4. WRITING STYLE ADHERENCE
   - Is the chapter following the specified writing style guidance?
   - Are there areas where the style could be better maintained?

Writing style guidance to maintain:
"""
        style_section += writing_style + "\n"
        
        writing_style_section = "### Writing Style Adherence Issues\n[List specific examples where the writing style guidance is not being followed]\n"
    
    # Replace placeholders
    prompt = template.replace("{{CHAPTER_NUMBER}}", str(chapter_num))
    prompt = prompt.replace("{{CHAPTER_TITLE}}", chapter_title)
    prompt = prompt.replace("{{CHAPTER_DESCRIPTION}}", chapter_description)
    prompt = prompt.replace("{{PREVIOUS_CONTEXT}}", previous_context)
    prompt = prompt.replace("{{DRAFT_CHAPTER}}", draft_chapter)
    prompt = prompt.replace("{{STYLE_SECTION}}", style_section)
    prompt = prompt.replace("{{WRITING_STYLE_SECTION}}", writing_style_section)
    prompt = prompt.replace("{{SEED_CONTENT}}", seed_content if seed_content else "")
    prompt = prompt.replace("{{GENRE}}", genre)
    
    # Save the prompt for reference - now under stories/<story>/prompts directory
    if story_title:
        story_name = story_title.lower().replace(" ", "_")
    else:
        story_name = "unnamed_story"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get parent directory from context (assumes this is called during story generation)
    prompts_dir = Path("stories") / story_name / "prompts"
    os.makedirs(prompts_dir, exist_ok=True)
    
    prompt_path = prompts_dir / f"{story_name}_ch{chapter_num:02d}_consistency_critique_{timestamp}.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"  Saved consistency critique prompt to {prompt_path}")
    
    try:
        start_time = time.time()
        
        # Define a function to call Gemini and process the response
        def call_gemini_for_critique():
            # Configure the API key at the client level
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = GenerativeModel(
                model_name="gemini-2.5-pro-exp-03-25"
            )
            
            response_text = ""
            for chunk in model.generate_content(prompt, stream=True):
                response_text += chunk.text
                # Print a progress indicator
                print(".", end="", flush=True)
            
            print()  # New line after progress dots
            
            return response_text
        
        # Call with retry logic
        response = retry_with_exponential_backoff(call_gemini_for_critique)
        
        # Calculate and display generation time
        elapsed_time = time.time() - start_time
        print(f"Consistency critique completed in {elapsed_time:.1f} seconds")
        
        # Extract the critique text
        critique_text = extract_message_content(response)
        
        # Save Gemini's response to the responses directory
        if story_title:
            story_name = story_title.lower().replace(" ", "_")
        else:
            story_name = "unnamed_story"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        responses_dir = Path("stories") / story_name / "responses"
        os.makedirs(responses_dir, exist_ok=True)
        response_path = responses_dir / f"{story_name}_ch{chapter_num:02d}_consistency_critique_{timestamp}.txt"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(critique_text)
        print(f"  Saved consistency critique response to {response_path}")
        
        # Extract content between <critique> tags if present
        import re
        pattern = r'<critique>(.*?)</critique>'
        match = re.search(pattern, critique_text, re.DOTALL)
        if match:
            critique = match.group(1).strip()
        else:
            # If tags not found, use the whole response
            critique = critique_text
        
        return critique
        
    except Exception as e:
        print(f"Error in consistency critique: {e}")
        # Return a minimal critique if the operation fails
        return f"Unable to generate a detailed critique due to an error: {e}"

def revise_for_consistency(
    draft_chapter: str,
    chapter_num: int,
    chapter_title: str,
    chapter_description: str,
    story_title: str,
    previous_chapters: List[str] = None,
    max_context_chars: int = 400000,  # Allow a large context for consistency checking
    writing_style: str = None,
    seed_content: str = None,
    genre: str = "fiction"
) -> str:
    """Revise chapter for narrative consistency based on a detailed critique.
    
    This is a two-step process:
    1. Generate a consistency critique
    2. Revise the chapter based on the critique
    """
    
    # Step 1: Generate consistency critique
    consistency_critique = critique_for_consistency(
        draft_chapter=draft_chapter,
        chapter_num=chapter_num,
        chapter_title=chapter_title,
        chapter_description=chapter_description,
        story_title=story_title,
        previous_chapters=previous_chapters,
        max_context_chars=max_context_chars,
        writing_style=writing_style,
        seed_content=seed_content,
        genre=genre
    )
    
    print(f"Revising Chapter {chapter_num} based on consistency critique...")
    
    # Define story name from story title
    if story_title:
        story_name = story_title.lower().replace(" ", "_")
    else:
        story_name = "unnamed_story"
        
    # Define source and destination template paths
    source_template_path = Path("simple_prompts/templates/revise_consistency.txt")
    template_path = Path("stories") / story_name / "prompts" / "templates" / "revise_consistency.txt"
    
    # Create templates directory
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    
    # Copy the template from source
    if os.path.exists(source_template_path):
        import shutil
        shutil.copy2(source_template_path, template_path)
    else:
        print(f"Error: Source template {source_template_path} not found.")
        sys.exit(1)
    
    # Load the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Prepare previous context if it exists
    previous_context = ""
    if previous_chapters:
        previous_context = "Previous chapters:\n\n"
        for i, chapter_text in enumerate(previous_chapters):
            # Use the most recent chapter numbers
            chapter_number = chapter_num - len(previous_chapters) + i
            if chapter_number > 0:
                previous_context += f"<chapter_{chapter_number}>\n{chapter_text}\n</chapter_{chapter_number}>\n\n"
    
    # Replace placeholders
    prompt = template.replace("{{DRAFT_CHAPTER}}", draft_chapter)
    prompt = prompt.replace("{{SEED_TEXT}}", f"Title: {story_title}\nChapter {chapter_num}: {chapter_title}\n\nChapter Description:\n{chapter_description}")
    prompt = prompt.replace("{{CRITIQUE}}", consistency_critique)
    prompt = prompt.replace("{{PREVIOUS_CONTEXT}}", previous_context)
    prompt = prompt.replace("{{SEED_CONTENT}}", seed_content if seed_content else "")
    prompt = prompt.replace("{{GENRE}}", genre)
    
    # Save the prompt for reference - now under stories/<story>/prompts directory
    if story_title:
        story_name = story_title.lower().replace(" ", "_")
    else:
        story_name = "unnamed_story"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get parent directory from context (assumes this is called during story generation)
    prompts_dir = Path("stories") / story_name / "prompts"
    os.makedirs(prompts_dir, exist_ok=True)
    
    prompt_path = prompts_dir / f"{story_name}_ch{chapter_num:02d}_consistency_revision_{timestamp}.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"  Saved consistency revision prompt to {prompt_path}")
    
    try:
        start_time = time.time()
        
        # Define a function to call Gemini and process the response
        def call_gemini_for_revision():
            # Configure the API key at the client level
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = GenerativeModel(
                model_name="gemini-2.5-pro-exp-03-25"
            )
            
            response_text = ""
            for chunk in model.generate_content(prompt, stream=True):
                response_text += chunk.text
                # Print a progress indicator
                print(".", end="", flush=True)
            
            print()  # New line after progress dots
            
            return response_text
        
        # Call with retry logic
        response = retry_with_exponential_backoff(call_gemini_for_revision)
        
        # Calculate and display generation time
        elapsed_time = time.time() - start_time
        print(f"Consistency revision completed in {elapsed_time:.1f} seconds")
        
        # Extract the revised chapter text
        revised_text = extract_message_content(response)
        
        # Save Gemini's response to the responses directory
        if story_title:
            story_name = story_title.lower().replace(" ", "_")
        else:
            story_name = "unnamed_story"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        responses_dir = Path("stories") / story_name / "responses"
        os.makedirs(responses_dir, exist_ok=True)
        response_path = responses_dir / f"{story_name}_ch{chapter_num:02d}_consistency_revision_{timestamp}.txt"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(revised_text)
        print(f"  Saved consistency revision response to {response_path}")
        
        # Extract content between <chapter> tags if present
        import re
        pattern = r'<chapter>(.*?)</chapter>'
        match = re.search(pattern, revised_text, re.DOTALL)
        if match:
            revised_chapter = match.group(1).strip()
        else:
            # If tags not found, use the whole response
            revised_chapter = revised_text
            
        return revised_chapter
        
    except Exception as e:
        print(f"Error in consistency revision: {e}")
        # Return original draft if revision fails
        return draft_chapter


def critique_for_style(
    chapter_content: str,
    chapter_num: int,
    chapter_title: str,
    story_title: str = "",
    writing_style: str = None,
    seed_content: str = None,
    genre: str = "fiction"
) -> str:
    """Critique chapter for writing style, prose quality, and readability."""
    
    print(f"Critiquing Chapter {chapter_num} for writing style and prose quality...")
    
    # Define story name from story title
    if story_title:
        story_name = story_title.lower().replace(" ", "_")
    else:
        story_name = "unnamed_story"
        
    # Define source and destination template paths
    source_template_path = Path("simple_prompts/templates/critique_chapter.txt")
    template_path = Path("stories") / story_name / "prompts" / "templates" / "critique_chapter.txt"
    
    # Create templates directory
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    
    # Copy the template from source
    if os.path.exists(source_template_path):
        import shutil
        shutil.copy2(source_template_path, template_path)
    else:
        print(f"Error: Source template {source_template_path} not found.")
        sys.exit(1)
    
    # Load the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Get previous chapter if available (for context)
    previous_chapter_text = ""
    
    # Replace placeholders
    prompt = template.replace("{{DRAFT_CHAPTER}}", chapter_content)
    prompt = prompt.replace("{{SEED_TEXT}}", f"Title: {story_title}\nChapter {chapter_num}: {chapter_title}")
    prompt = prompt.replace("{{PREVIOUS_CHAPTER}}", previous_chapter_text)
    prompt = prompt.replace("{{STORY_TITLE}}", story_title if story_title else "")
    prompt = prompt.replace("{{CHAPTER_NUMBER}}", str(chapter_num))
    prompt = prompt.replace("{{SEED_CONTENT}}", seed_content if seed_content else "")
    prompt = prompt.replace("{{GENRE}}", genre)
    
    # Save the prompt for reference - now under stories/<story>/prompts directory
    if story_title:
        story_name = story_title.lower().replace(" ", "_")
    else:
        story_name = "unnamed_story"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get parent directory from context (assumes this is called during story generation)
    prompts_dir = Path("stories") / story_name / "prompts"
    os.makedirs(prompts_dir, exist_ok=True)
    
    prompt_path = prompts_dir / f"{story_name}_ch{chapter_num:02d}_style_critique_{timestamp}.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"  Saved style critique prompt to {prompt_path}")
    
    try:
        start_time = time.time()
        
        # Define a function to call Gemini and process the response
        def call_gemini_for_style_critique():
            # Configure the API key at the client level
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = GenerativeModel(
                model_name="gemini-2.5-pro-exp-03-25"
            )
            
            response_text = ""
            for chunk in model.generate_content(prompt, stream=True):
                response_text += chunk.text
                # Print a progress indicator
                print(".", end="", flush=True)
            
            print()  # New line after progress dots
            
            return response_text
        
        # Call with retry logic
        response = retry_with_exponential_backoff(call_gemini_for_style_critique)
        
        # Calculate and display generation time
        elapsed_time = time.time() - start_time
        print(f"Style critique completed in {elapsed_time:.1f} seconds")
        
        # Extract the critique text
        critique_text = extract_message_content(response)
        
        # Save Gemini's response to the responses directory
        if story_title:
            story_name = story_title.lower().replace(" ", "_")
        else:
            story_name = "unnamed_story"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        responses_dir = Path("stories") / story_name / "responses"
        os.makedirs(responses_dir, exist_ok=True)
        response_path = responses_dir / f"{story_name}_ch{chapter_num:02d}_style_critique_{timestamp}.txt"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(critique_text)
        print(f"  Saved style critique response to {response_path}")
        
        # Extract content between <critique> tags if present
        import re
        pattern = r'<critique>(.*?)</critique>'
        match = re.search(pattern, critique_text, re.DOTALL)
        if match:
            critique = match.group(1).strip()
        else:
            # If tags not found, use the whole response
            critique = critique_text
        
        return critique
        
    except Exception as e:
        print(f"Error in style critique: {e}")
        # Return a minimal critique if the operation fails
        return f"Style and prose need improvement. Focus on sentence variety, word choice, and showing rather than telling."

def revise_for_style(
    chapter_content: str,
    chapter_num: int,
    chapter_title: str,
    story_title: str = "",
    writing_style: str = None,
    seed_content: str = None,
    genre: str = "fiction"
) -> str:
    """Revise chapter for writing style based on a detailed critique.
    
    This is a two-step process:
    1. Generate a style critique
    2. Revise the chapter based on the critique
    """
    
    # Step 1: Generate style critique
    style_critique = critique_for_style(
        chapter_content=chapter_content,
        chapter_num=chapter_num,
        chapter_title=chapter_title,
        story_title=story_title,
        writing_style=writing_style,
        seed_content=seed_content,
        genre=genre
    )
    
    print(f"Revising Chapter {chapter_num} based on style critique...")
    
    # Define story name from story title
    if story_title:
        story_name = story_title.lower().replace(" ", "_")
    else:
        story_name = "unnamed_story"
        
    # Define source and destination template paths
    source_template_path = Path("simple_prompts/templates/revise_chapter.txt")
    template_path = Path("stories") / story_name / "prompts" / "templates" / "revise_chapter.txt"
    
    # Create templates directory
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    
    # Copy the template from source
    if os.path.exists(source_template_path):
        import shutil
        shutil.copy2(source_template_path, template_path)
    else:
        print(f"Error: Source template {source_template_path} not found.")
        sys.exit(1)
    
    # Load the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders
    prompt = template.replace("{{DRAFT_CHAPTER}}", chapter_content)
    prompt = prompt.replace("{{SEED_TEXT}}", f"Title: {story_title}\nChapter {chapter_num}: {chapter_title}")
    prompt = prompt.replace("{{CRITIQUE}}", style_critique)
    prompt = prompt.replace("{{STORY_TITLE}}", story_title if story_title else "")
    prompt = prompt.replace("{{CHAPTER_NUMBER}}", str(chapter_num))
    prompt = prompt.replace("{{SEED_CONTENT}}", seed_content if seed_content else "")
    prompt = prompt.replace("{{GENRE}}", genre)
    
    # Save the prompt for reference - now under stories/<story>/prompts directory
    if story_title:
        story_name = story_title.lower().replace(" ", "_")
    else:
        story_name = "unnamed_story"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get parent directory from context (assumes this is called during story generation)
    prompts_dir = Path("stories") / story_name / "prompts"
    os.makedirs(prompts_dir, exist_ok=True)
    
    prompt_path = prompts_dir / f"{story_name}_ch{chapter_num:02d}_style_revision_{timestamp}.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"  Saved style revision prompt to {prompt_path}")
    
    try:
        start_time = time.time()
        
        # Define a function to call Gemini and process the response
        def call_gemini_for_style_revision():
            # Configure the API key at the client level
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = GenerativeModel(
                model_name="gemini-2.5-pro-exp-03-25"
            )
            
            response_text = ""
            for chunk in model.generate_content(prompt, stream=True):
                response_text += chunk.text
                # Print a progress indicator
                print(".", end="", flush=True)
            
            print()  # New line after progress dots
            
            return response_text
        
        # Call with retry logic
        response = retry_with_exponential_backoff(call_gemini_for_style_revision)
        
        # Calculate and display generation time
        elapsed_time = time.time() - start_time
        print(f"Style revision completed in {elapsed_time:.1f} seconds")
        
        # Extract the revised chapter text
        revised_text = extract_message_content(response)
        
        # Save Gemini's response to the responses directory
        if story_title:
            story_name = story_title.lower().replace(" ", "_")
        else:
            story_name = "unnamed_story"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        responses_dir = Path("stories") / story_name / "responses"
        os.makedirs(responses_dir, exist_ok=True)
        response_path = responses_dir / f"{story_name}_ch{chapter_num:02d}_style_revision_{timestamp}.txt"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(revised_text)
        print(f"  Saved style revision response to {response_path}")
        
        # Extract content between <chapter> tags if present
        import re
        pattern = r'<chapter>(.*?)</chapter>'
        match = re.search(pattern, revised_text, re.DOTALL)
        if match:
            revised_chapter = match.group(1).strip()
        else:
            # If tags not found, use the whole response
            revised_chapter = revised_text
            
        return revised_chapter
        
    except Exception as e:
        print(f"Error in style revision: {e}")
        # Return unrevised content if revision fails
        return chapter_content

def extract_writing_style(seed_path: str, genre: str = "fiction") -> str:
    """Extract writing style guidance from the seed file."""
    print(f"Extracting writing style guidance from seed file: {seed_path}")
    
    # Read the seed file
    try:
        with open(seed_path, 'r') as file:
            seed_content = file.read()
    except Exception as e:
        print(f"Error reading seed file: {e}")
        sys.exit(1)
    
    # Extract story title from seed filename
    seed_filename = Path(seed_path).stem
    story_title = seed_filename.replace('-', ' ').replace('_', ' ').title()
    
    # Load the template for writing style extraction
    template_path = Path("simple_prompts/templates/extract_writing_style.txt")
    
    # Check if template exists, if not create an empty file to avoid errors
    if not os.path.exists(template_path):
        print(f"Warning: Template {template_path} not found.")
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        with open(template_path, 'w') as f:
            f.write("Template file not found")
        return "Could not extract writing style due to missing template."
    
    # Load the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders
    prompt = template.replace("{{SEED_CONTENT}}", seed_content)
    prompt = prompt.replace("{{STORY_TITLE}}", story_title)
    prompt = prompt.replace("{{GENRE}}", genre)
    
    try:
        print("Generating writing style guidance...")
        
        # Define a function to call Gemini and process the response
        def call_gemini_for_writing_style():
            # Configure the API key at the client level
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = GenerativeModel(
                model_name="gemini-2.5-pro-exp-03-25"
            )
            
            response_text = ""
            for chunk in model.generate_content(prompt, stream=True):
                response_text += chunk.text
                # Print a progress indicator
                print(".", end="", flush=True)
            
            print()  # New line after progress dots
            
            return response_text
        
        # Call with retry logic
        response = retry_with_exponential_backoff(call_gemini_for_writing_style)
        
        # Extract the writing style guidance text
        writing_style_guidance = extract_message_content(response)
        
        # Extract seed filename without extension
        seed_filename = Path(seed_path).stem
        
        # Save writing style guidance to both the global prompts directory and the story-specific directory
        # First, save to global prompts directory for reference
        global_path = Path("prompts") / f"{seed_filename}_writing_style.txt"
        os.makedirs(os.path.dirname(global_path), exist_ok=True)
        with open(global_path, 'w') as file:
            file.write(writing_style_guidance)
        
        # Then save to the story-specific directory that will be created
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        story_dir = Path("stories") / seed_filename.lower()
        os.makedirs(story_dir, exist_ok=True)
            
        print(f"Writing style guidance saved to {global_path}")
        
        return writing_style_guidance
        
    except Exception as e:
        print(f"Error extracting writing style: {e}")
        print("Failed to extract writing style. Please create a writing style guide manually.")
        # Return a basic style guide
        return """## Basic Writing Style Guide

### Style Parameters
TONE: Balanced and natural, shifting appropriately with the narrative
PERSPECTIVE: Third person limited
SENTENCE STRUCTURE: Mix of simple and complex sentences for rhythm
DIALOGUE STYLE: Natural, character-appropriate dialogue with minimal tags
DESCRIPTION LEVEL: Clear, vivid descriptions focusing on important elements
PACING: Varied according to scene needs
LANGUAGE PREFERENCES: Clear, accessible language avoiding excessive ornamentation
"""

def generate_story(
    seed_path: str, 
    output_dir: str = "stories",
    temperature: float = 1.0, 
    max_context_chars: int = 500000, 
    skip_revisions: bool = False, 
    revision_context: int = 400000, 
    num_chapters: int = 25, 
    writing_style: str = None, 
    genre: str = "fiction", 
    seed_content: str = None
):
    """Generate story chapters directly from seed content."""
    # Gemini client is initialized with genai.configure() in main()
    
    # Get story title from seed filename
    seed_filename = Path(seed_path).stem
    story_title = seed_filename.replace('-', ' ').replace('_', ' ').title()
    
    # Read seed content if not provided
    if seed_content is None and seed_path:
        try:
            with open(seed_path, 'r') as file:
                seed_content = file.read()
            print(f"Read seed content from {seed_path}")
        except Exception as e:
            print(f"Warning: Could not read seed file: {e}")
            seed_content = None
    
    # If no writing style was provided, extract it from the seed file
    if writing_style is None and seed_path:
        writing_style = extract_writing_style(seed_path, genre=genre)
    
    # Create story directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    story_dir = Path(output_dir) / seed_filename.lower() / timestamp
    setup_directories(story_dir)
    
    # Save seed content to the story directory
    if seed_content:
        seed_path = story_dir / "seed.md"
        with open(seed_path, 'w') as file:
            file.write(seed_content)
        print(f"Seed content saved to {seed_path}")
    
    # Save writing style to the story directory if available
    if writing_style:
        writing_style_path = story_dir / "writing_style.txt"
        with open(writing_style_path, 'w') as file:
            file.write(writing_style)
        print(f"Writing style saved to {writing_style_path}")
    
    # Create metadata file
    metadata = {
        "title": story_title,
        "generated_at": datetime.now().isoformat(),
        "num_chapters": num_chapters,
        "writing_style_available": writing_style is not None,
        "genre": genre
    }
    with open(story_dir / "metadata.json", 'w') as file:
        json.dump(metadata, file, indent=2)
    
    # Create a combined story file
    combined_story = f"# {story_title}\n\n"
    
    # Track previous chapters to provide as context
    previous_chapters = []
    current_context_size = 0
    
    # Generate each chapter
    for chapter_num in range(1, num_chapters + 1):
        chapter_file = f"chapter_{chapter_num:02d}.md"
        chapter_path = story_dir / "chapters" / chapter_file
        
        # Generate chapter content with Gemini including previous chapters as context
        chapter_content = generate_chapter(
            chapter_num=chapter_num,
            story_title=story_title,
            previous_chapters=previous_chapters,
            seed_content=seed_content,
            temperature=temperature,
            skip_revisions=skip_revisions,
            revision_context=revision_context,
            writing_style=writing_style,
            genre=genre
        )
        
        # Save chapter to file
        with open(chapter_path, 'w') as file:
            file.write(chapter_content)
            
        print(f"Saved chapter to {chapter_path}")
        
        # Add to combined story
        combined_story += f"{chapter_content}\n\n"
        
        # Add the current chapter to previous chapters for context in next generation
        previous_chapters.append(chapter_content)
        
        # Update context size and manage context window
        current_context_size += len(chapter_content)
        
        # If context size exceeds max, remove oldest chapters until we're under the limit
        while current_context_size > max_context_chars and len(previous_chapters) > 1:
            removed_chapter = previous_chapters.pop(0)
            current_context_size -= len(removed_chapter)
            print(f"Removed oldest chapter from context to stay within context limit")
        
        print(f"Context size: {current_context_size} characters, {len(previous_chapters)} chapters in context")
        
        # Sleep briefly to avoid API rate limits
        time.sleep(1)
    
    # Save the combined story file
    combined_file = story_dir / "chapters" / f"{story_title.replace(' ', '_').lower()}.md"
    with open(combined_file, 'w') as file:
        file.write(combined_story)
    
    print(f"Story generation complete. Files saved to {story_dir}/")
    print(f"Combined story saved to {combined_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a story from a seed file")
    parser.add_argument("--seed", required=True, help="Path to the seed markdown file")
    parser.add_argument("--num-chapters", type=int, required=True, help="Number of chapters to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--max-context", type=int, default=500000, help="Maximum size in characters for previous chapters context (default: 500000)")
    parser.add_argument("--skip-revisions", action="store_true", help="Skip the consistency and style revision steps (faster but lower quality)")
    parser.add_argument("--revision-context", type=int, default=400000, help="Maximum context size for consistency revision (default: 400000)")
    parser.add_argument("--genre", type=str, default="fiction", help="Genre of the story (default: fiction)")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Ensure stories directory exists
    Path("stories").mkdir(exist_ok=True)
    # Ensure prompts directory exists for writing style storage
    Path("prompts").mkdir(exist_ok=True)
    
    # Try to load .env file manually if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded .env file using python-dotenv")
    except ImportError:
        print("python-dotenv not installed, attempting to read .env file manually")
        try:
            with open(".env") as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            print("Manually loaded .env file")
        except Exception as e:
            print(f"Error loading .env file manually: {e}")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"API key: {api_key[:4]}...{api_key[-4:] if api_key else None}")
    
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found")
        sys.exit(1)
    
    # No need to initialize the client globally anymore
    # We'll pass api_key to each GenerativeModel instance
    
    # Extract writing style guidance
    writing_style = extract_writing_style(args.seed, genre=args.genre)
    
    # Read the seed content
    try:
        with open(args.seed, 'r') as file:
            seed_content = file.read()
    except Exception as e:
        print(f"Error reading seed file: {e}")
        sys.exit(1)
    
    print(f"Will generate {args.num_chapters} chapters")
    
    # Generate the story with maximum context size for previous chapters
    generate_story(
        seed_path=args.seed,
        output_dir="stories",
        temperature=args.temperature,
        max_context_chars=args.max_context,
        skip_revisions=args.skip_revisions,
        revision_context=args.revision_context,
        num_chapters=args.num_chapters,
        writing_style=writing_style,
        genre=args.genre,
        seed_content=seed_content
    )

def generate():
    # Configure the Gemini API
    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = GenerativeModel(
        model_name="gemini-2.5-pro-exp-03-25"
    )
    
    prompt = """INSERT_INPUT_HERE"""
    
    for chunk in model.generate_content(prompt, stream=True):
        print(chunk.text, end="")

if __name__ == "__main__":
    main()