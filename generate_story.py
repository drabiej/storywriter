#!/usr/bin/env python3
"""
Generate a story from an outline using Claude AI.

This script takes an outline file and generates a complete story chapter by chapter,
with each chapter going through a two-step process: generation and revision.
The revision step ensures the content strictly conforms to the outline, seed, 
character profiles, and hooks, while maintaining story continuity and realism.

It incorporates character profiles and hooks to maintain story continuity and engagement.
It can handle both outlines with explicit chapter markers and those without.
For outlines without explicit chapter markers, it will intelligently divide the content
into chapter-sized segments based on section headers or paragraph boundaries.
"""

import os
import sys
import argparse
import time
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import anthropic

def extract_message_content(response) -> str:
    """Extract text content from an Anthropic response message.
    
    This function handles different response structures, including thinking blocks.
    It prioritizes finding text blocks in the response content.
    """
    # Check for text blocks first, they're what we want
    content_text = None
    for block in response.content:
        if not hasattr(block, 'type') or block.type == "text":
            if hasattr(block, 'text'):
                content_text = block.text
                break
    
    # If no text block found, fall back to using first block or error message
    if not content_text:
        content_block = response.content[0]
        if hasattr(content_block, 'text'):
            content_text = content_block.text
        else:
            content_text = "Unable to extract content properly from response."
            
    return content_text

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a story from an outline using Claude")
    
    # Story directory argument (new primary method)
    parser.add_argument(
        "--story-dir", 
        type=str, 
        help="Path to the story directory containing all story files (created by generate_outline.py)"
    )
    
    # Legacy individual file arguments
    parser.add_argument(
        "--outline", 
        type=str, 
        help="Path to the outline file (use --story-dir instead for consolidated files)"
    )
    parser.add_argument(
        "--hooks",
        type=str,
        help="Path to the hooks file (use --story-dir instead for consolidated files)"
    )
    parser.add_argument(
        "--profiles",
        type=str,
        help="Path to the character profiles file (use --story-dir instead for consolidated files)"
    )
    parser.add_argument(
        "--seed",
        type=str,
        help="Path to the original seed file (use --story-dir instead for consolidated files)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing all input files (use --story-dir instead for consolidated files)"
    )
    
    # Other arguments
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path (default: <story_dir>/<story_name>.md)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0, 
        help="Temperature for Claude generation (default: 1.0)"
    )
    parser.add_argument(
        "--chapters", 
        type=int, 
        help="Number of chapters to generate (default: all from outline)"
    )
    parser.add_argument(
        "--target-chapters",
        type=int,
        help="Target number of chapters to divide the outline into (used only for outlines without explicit chapter markers)"
    )
    parser.add_argument(
        "--start-chapter",
        type=int,
        default=1,
        help="Chapter to start generation from (default: 1)"
    )
    parser.add_argument(
        "--no-revision",
        action="store_true",
        help="Skip the revision step (faster but lower quality)"
    )
    parser.add_argument(
        "--revision-temperature",
        type=float,
        default=1.0,
        help="Temperature for revision step (default: 1.0)"
    )
    return parser.parse_args()

def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def find_story_files(
    story_dir: Optional[str] = None,
    outline_path: Optional[str] = None, 
    input_dir: Optional[str] = None, 
    hooks_path: Optional[str] = None, 
    profiles_path: Optional[str] = None,
    seed_path: Optional[str] = None
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Find related story files based on story directory or outline name."""
    
    # If story_dir is provided, that's our new primary method
    if story_dir:
        story_dir_path = Path(story_dir)
        if not story_dir_path.exists() or not story_dir_path.is_dir():
            print(f"Warning: Story directory '{story_dir}' not found or is not a directory")
            if outline_path:
                print(f"Falling back to using outline: {outline_path}")
            else:
                raise ValueError(f"Story directory '{story_dir}' not found and no outline specified")
        else:
            # Check for metadata file first
            metadata_path = story_dir_path / "story_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    found_outline = metadata.get("outline")
                    found_hooks = metadata.get("hooks")
                    found_profiles = metadata.get("profiles")
                    found_seed = metadata.get("seed")
                    
                    # Validate paths exist
                    if found_outline and Path(found_outline).exists():
                        print(f"Found outline file from metadata: {found_outline}")
                    else:
                        found_outline = None
                        
                    if found_hooks and Path(found_hooks).exists():
                        print(f"Found hooks file from metadata: {found_hooks}")
                    else:
                        found_hooks = None
                        
                    if found_profiles and Path(found_profiles).exists():
                        print(f"Found profiles file from metadata: {found_profiles}")
                    else:
                        found_profiles = None
                        
                    if found_seed and Path(found_seed).exists():
                        print(f"Found seed file from metadata: {found_seed}")
                    else:
                        found_seed = None
                    
                    # If we found at least the outline, return the files
                    if found_outline:
                        return found_outline, found_hooks, found_profiles, found_seed
                    
                except Exception as e:
                    print(f"Error reading metadata file: {e}")
                    print("Falling back to file pattern search")
            
            # If metadata file not found or invalid, search by pattern in the story directory
            dir_name = story_dir_path.name
            
            # Look for outline file
            outline_pattern = f"{dir_name}_outline.md"
            outline_files = list(story_dir_path.glob(outline_pattern))
            if not outline_files:
                # Try any file with "outline" in the name
                outline_files = list(story_dir_path.glob("*outline*.md"))
            
            if outline_files:
                found_outline = str(outline_files[0])
                print(f"Found outline file: {found_outline}")
            else:
                if outline_path:
                    found_outline = outline_path
                    print(f"Using provided outline file: {found_outline}")
                else:
                    raise ValueError(f"No outline file found in story directory: {story_dir}")
            
            # Look for hooks file
            hooks_pattern = f"{dir_name}_hooks.md"
            hooks_files = list(story_dir_path.glob(hooks_pattern))
            found_hooks = str(hooks_files[0]) if hooks_files else None
            if found_hooks:
                print(f"Found hooks file: {found_hooks}")
            
            # Look for profiles file
            profiles_pattern = f"{dir_name}_profiles.md"
            profiles_files = list(story_dir_path.glob(profiles_pattern))
            found_profiles = str(profiles_files[0]) if profiles_files else None
            if found_profiles:
                print(f"Found profiles file: {found_profiles}")
            
            # Look for seed file
            seed_pattern = f"{dir_name}_seed.md"
            seed_files = list(story_dir_path.glob(seed_pattern))
            found_seed = str(seed_files[0]) if seed_files else None
            if found_seed:
                print(f"Found seed file: {found_seed}")
            
            return found_outline, found_hooks, found_profiles, found_seed
    
    # If we reach here, we're using the legacy method with outline_path as primary
    if not outline_path:
        raise ValueError("Either --story-dir or --outline must be provided")
    
    outline_file = Path(outline_path)
    story_name = outline_file.stem
    
    # Strip timestamp if present (e.g., england_20250311_015444.md -> england)
    base_name = story_name.split('_')[0] if '_' in story_name else story_name
    
    found_hooks = None
    found_profiles = None
    found_seed = None
    
    # If files are explicitly provided, use those
    if hooks_path:
        if os.path.exists(hooks_path):
            found_hooks = hooks_path
        else:
            print(f"Warning: Specified hooks file '{hooks_path}' not found")
    
    if profiles_path:
        if os.path.exists(profiles_path):
            found_profiles = profiles_path
        else:
            print(f"Warning: Specified profiles file '{profiles_path}' not found")
    
    if seed_path:
        if os.path.exists(seed_path):
            found_seed = seed_path
        else:
            print(f"Warning: Specified seed file '{seed_path}' not found")
            
    # If input_dir is provided, look for files there
    if input_dir:
        input_dir_path = Path(input_dir)
        
        # Check for hooks file
        if not found_hooks:
            hooks_pattern = f"{base_name}_hooks*.md"
            hooks_files = list(input_dir_path.glob(hooks_pattern))
            if hooks_files:
                # Use the most recent one based on filename timestamp
                hooks_files.sort(reverse=True)
                found_hooks = str(hooks_files[0])
                print(f"Found hooks file: {found_hooks}")
        
        # Check for profiles file
        if not found_profiles:
            profiles_pattern = f"{base_name}_profiles*.md"
            profiles_files = list(input_dir_path.glob(profiles_pattern))
            if profiles_files:
                # Use the most recent one based on filename timestamp
                profiles_files.sort(reverse=True)
                found_profiles = str(profiles_files[0])
                print(f"Found profiles file: {found_profiles}")
        
        # Check for seed file in input directory
        if not found_seed:
            seed_pattern = f"{base_name}*seed*.md"
            seed_files = list(input_dir_path.glob(seed_pattern))
            if not seed_files:
                seed_pattern = f"{base_name}.md"
                seed_files = list(input_dir_path.glob(seed_pattern))
            if seed_files:
                found_seed = str(seed_files[0])
                print(f"Found seed file: {found_seed}")
    
    # If still not found, look in standard directories
    if not found_hooks:
        # Look in hooks directory
        hooks_pattern = f"hooks/{base_name}_hooks_*.md"
        hooks_files = glob.glob(hooks_pattern)
        if hooks_files:
            hooks_files.sort(reverse=True)
            found_hooks = hooks_files[0]
            print(f"Found hooks file: {found_hooks}")
    
    if not found_profiles:
        # Look in profiles directory
        profiles_pattern = f"profiles/{base_name}_profiles_*.md"
        profiles_files = glob.glob(profiles_pattern)
        if profiles_files:
            profiles_files.sort(reverse=True)
            found_profiles = profiles_files[0]
            print(f"Found profiles file: {found_profiles}")
    
    if not found_seed:
        # Look in seeds directory
        seed_pattern = f"seeds/{base_name}.md"
        seed_files = glob.glob(seed_pattern)
        if seed_files:
            found_seed = seed_files[0]
            print(f"Found seed file: {found_seed}")
    
    return outline_path, found_hooks, found_profiles, found_seed

def parse_outline(outline_text: str, target_chapters: int = None) -> List[Dict[str, str]]:
    """Parse the outline into chapters with titles and descriptions.
    
    If the outline has explicit chapter markers (## Chapter or ## heading), use those.
    Otherwise, intelligently divide the content into chapter-sized segments.
    
    Args:
        outline_text: The full outline text
        target_chapters: Optional number of chapters to aim for when auto-dividing
        
    Returns:
        A list of chapter dictionaries with 'title' and 'description' keys
    """
    # First, check if there are explicit chapter markers - match any format that contains CHAPTER
    lines = outline_text.split("\n")
    has_chapter_markers = any(("CHAPTER" in line or "Chapter" in line) and line.startswith("#") 
                             for line in lines)
    
    # If explicit chapter markers exist, use traditional parsing
    if has_chapter_markers:
        chapters = []
        current_chapter = None
        
        for line in lines:
            if line.startswith("# "):
                # This is the story title, skip it
                continue
            # Match any heading format that includes the word "CHAPTER" or "Chapter" in the line
            elif ("CHAPTER" in line or "Chapter" in line) and line.startswith("#"):
                if current_chapter:
                    chapters.append(current_chapter)
                chapter_title = line.lstrip("#").strip()
                current_chapter = {"title": chapter_title, "description": ""}
            elif current_chapter:
                current_chapter["description"] += line + "\n"
        
        if current_chapter:
            chapters.append(current_chapter)
        
        return chapters
    
    # If no explicit chapter markers, divide content intelligently
    print("No explicit chapter markers found. Auto-dividing outline into chapters...")
    
    # Extract the story title if it exists
    story_title = ""
    for line in lines:
        if line.startswith("# "):
            story_title = line.lstrip("# ").strip()
            break
    
    # Identify major section headers (### level headers)
    section_indices = []
    for i, line in enumerate(lines):
        if line.startswith("### "):
            section_indices.append(i)
    
    # If no section headers found or only a few, look for paragraph boundaries
    if len(section_indices) < 3:
        print("Insufficient section headers found. Using paragraph boundaries...")
        # Look for paragraph breaks (empty lines)
        section_indices = [i for i, line in enumerate(lines) if line.strip() == ""]
    
    # Determine how many chapters to create
    if target_chapters and target_chapters > 0:
        num_chapters = target_chapters
    else:
        # Default: aim for 8-12 chapters depending on content length
        content_length = len(outline_text)
        if content_length < 5000:
            num_chapters = 8
        elif content_length < 10000:
            num_chapters = 10
        else:
            num_chapters = 12
        print(f"Auto-dividing into approximately {num_chapters} chapters based on content length...")
    
    # Calculate number of sections per chapter to achieve target chapter count
    if len(section_indices) > 0:
        sections_per_chapter = max(1, len(section_indices) // num_chapters)
        
        # Create chapter boundaries
        chapter_boundaries = []
        for i in range(0, len(section_indices), sections_per_chapter):
            if i < len(section_indices):
                chapter_boundaries.append(section_indices[i])
        
        # Remove any boundaries that are too close to the start
        while len(chapter_boundaries) > 0 and chapter_boundaries[0] < 5:
            chapter_boundaries.pop(0)
            
        # Add the start of the content as the first boundary if needed
        if len(chapter_boundaries) == 0 or chapter_boundaries[0] > 5:
            # Find the first non-title, non-empty line
            start_idx = 0
            for i, line in enumerate(lines):
                if not line.startswith("#") and line.strip() != "":
                    start_idx = i
                    break
            chapter_boundaries.insert(0, start_idx)
            
        # Create chapters based on these boundaries
        chapters = []
        for i in range(len(chapter_boundaries)):
            start_idx = chapter_boundaries[i]
            end_idx = chapter_boundaries[i+1] if i+1 < len(chapter_boundaries) else len(lines)
            
            # Find a good title for this chapter
            chapter_content = "\n".join(lines[start_idx:end_idx])
            title = f"Chapter {i+1}"
            
            # Try to extract a meaningful title from content
            for line in lines[start_idx:min(start_idx+5, end_idx)]:
                if line.startswith("###"):
                    title = f"Chapter {i+1}: {line.lstrip('#').strip()}"
                    break
            
            chapters.append({
                "title": title,
                "description": chapter_content
            })
    else:
        # Fallback: divide evenly by line count if no structure found
        print("No clear structure found. Dividing evenly by line count...")
        lines_per_chapter = max(10, len(lines) // num_chapters)
        
        chapters = []
        for i in range(0, len(lines), lines_per_chapter):
            end_idx = min(i + lines_per_chapter, len(lines))
            chapter_content = "\n".join(lines[i:end_idx])
            chapters.append({
                "title": f"Chapter {len(chapters) + 1}",
                "description": chapter_content
            })
    
    print(f"Successfully divided outline into {len(chapters)} chapters.")
    return chapters

def generate_table_of_contents(story_content: str) -> str:
    """Generate a table of contents from the story content."""
    lines = story_content.split('\n')
    chapter_lines = []
    
    # Find all chapter headings
    for i, line in enumerate(lines):
        if line.startswith('## Chapter'):
            chapter_lines.append(line.strip())
    
    # Generate the TOC
    toc = "## Table of Contents\n\n"
    
    for i, chapter in enumerate(chapter_lines):
        # Remove the '##' prefix
        chapter_title = chapter[2:].strip()
        # Create a link-friendly version of the chapter title
        link = chapter_title.lower().replace(' ', '-').replace(':', '').replace(',', '')
        # Add the TOC entry
        toc += f"- [{chapter_title}](#{link})\n"
    
    return toc

def extract_chapter_hooks(hooks_text: str, chapter_number: int) -> Dict[str, str]:
    """Extract hooks relevant to the specific chapter."""
    if not hooks_text:
        return {}
    
    hooks_by_chapter = {}
    current_chapter = None
    current_content = []
    
    # Simple parsing to find chapter-specific hooks
    for line in hooks_text.split('\n'):
        if line.strip().startswith("### SECTION: Chapter"):
            # Save previous chapter's hooks if we were collecting any
            if current_chapter is not None and current_content:
                hooks_by_chapter[current_chapter] = '\n'.join(current_content)
                current_content = []
            
            # Extract chapter number - handle formats like "Chapter 3" or "Chapter 3 - Title"
            try:
                chapter_text = line.split("Chapter")[1].split()[0].split("-")[0].strip()
                current_chapter = int(chapter_text)
            except:
                current_chapter = None
        elif current_chapter is not None:
            current_content.append(line)
    
    # Save the last chapter's hooks
    if current_chapter is not None and current_content:
        hooks_by_chapter[current_chapter] = '\n'.join(current_content)
    
    # If no chapter-specific hooks found, return all hooks for context
    if not hooks_by_chapter:
        return {"all": hooks_text}
    
    # Return hooks for requested chapter and adjacent chapters
    result = {}
    for ch in range(chapter_number - 1, chapter_number + 2):
        if ch in hooks_by_chapter:
            result[ch] = hooks_by_chapter[ch]
    
    # If nothing found for this chapter range, return all hooks
    if not result:
        return {"all": hooks_text}
    
    return result

def generate_chapter_summary(
    client: anthropic.Anthropic,
    chapter_content: str,
    chapter_number: int,
    chapter_title: str,
    temperature: float = 0.7
) -> str:
    """Generate a concise bullet point summary of the key elements in a chapter."""
    
    print(f"Generating bullet point summary for Chapter {chapter_number}...", end="", flush=True)
    
    summary_prompt = f"""
    Please create a concise bullet point summary of Chapter {chapter_number}: {chapter_title}.
    
    Focus on:
    - Key plot developments
    - Important character actions and decisions
    - Critical revelations or information
    - Major emotional or relationship changes
    - Setting details that impact the storyline
    
    Format the summary as 5-10 bullet points, each no more than 1-2 sentences.
    Each bullet should capture one concrete, important element from the chapter.
    Be specific rather than general.
    
    The summary will be used to maintain continuity in future chapters, so focus on elements that future chapters need to build upon.
    """
    
    try:
        start_time = time.time()
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            temperature=temperature,
            messages=[
                {"role": "user", "content": f"Here is the content of Chapter {chapter_number}:\n\n{chapter_content}"},
                {"role": "user", "content": summary_prompt}
            ]
        )
        
        # Calculate and display generation time
        elapsed_time = time.time() - start_time
        print(f" done! ({elapsed_time:.1f} seconds)")
        
        # Extract the summary text
        summary_text = response.content[0].text
        
        return summary_text
        
    except Exception as e:
        print(f"\nError generating chapter summary: {e}")
        return f"• Chapter {chapter_number} summary generation failed"

def revise_chapter(
    client: anthropic.Anthropic,
    chapter_content: str,
    outline: str,
    chapter_info: Dict[str, str],
    chapter_number: int,
    hooks_text: Optional[str] = None,
    profiles_text: Optional[str] = None,
    seed_text: Optional[str] = None,
    previous_chapter: Optional[str] = None,
    temperature: float = 1.0
) -> str:
    """Revise a chapter to ensure it conforms to all requirements."""
    
    print(f"Revising Chapter {chapter_number}: {chapter_info['title']}...")
    print("Thinking and revising in progress...", end="", flush=True)
    
    # Prepare context for revision
    revision_context = []
    
    # Add the original seed if available
    if seed_text:
        revision_context.append({
            "role": "user",
            "content": f"Here is the original seed concept for the story:\n\n{seed_text}\n\nThis represents the initial vision for the story."
        })
    
    # Add the outline for context
    revision_context.append({
        "role": "user", 
        "content": f"Here is the detailed outline for the story:\n\n{outline}\n\nPlease reference this outline throughout the revision process."
    })
    
    # Add character profiles if available
    if profiles_text:
        revision_context.append({
            "role": "user",
            "content": f"Here are detailed character profiles for the story:\n\n{profiles_text}\n\nEnsure the chapter maintains consistent characterization according to these profiles."
        })
    
    # Add hooks if available
    if hooks_text:
        # Extract chapter-specific hooks
        chapter_hooks = extract_chapter_hooks(hooks_text, chapter_number)
        
        if 'all' in chapter_hooks:
            # If no chapter-specific hooks found, provide all hooks
            hook_content = f"Here are the strategic hooks for the story that ensure reader engagement:\n\n{chapter_hooks['all']}\n\nVerify these hooks are properly integrated in the chapter."
        else:
            # Format chapter-specific hooks
            hook_content = "Here are the strategic hooks relevant to this part of the story:\n\n"
            
            if chapter_number - 1 in chapter_hooks:
                hook_content += f"Hooks from previous chapter (Chapter {chapter_number-1}):\n{chapter_hooks[chapter_number-1]}\n\n"
            
            if chapter_number in chapter_hooks:
                hook_content += f"Hooks for this chapter (Chapter {chapter_number}):\n{chapter_hooks[chapter_number]}\n\n"
            
            if chapter_number + 1 in chapter_hooks:
                hook_content += f"Hooks for upcoming chapter (Chapter {chapter_number+1}):\n{chapter_hooks[chapter_number+1]}\n\n"
            
            hook_content += "Verify that these hooks are properly established, developed, or resolved in this chapter."
        
        revision_context.append({
            "role": "user",
            "content": hook_content
        })
    
    # Add previous chapter if available for continuity check
    if previous_chapter:
        revision_context.append({
            "role": "user",
            "content": f"Here is the previous chapter:\n\n{previous_chapter}\n\nEnsure this chapter maintains proper continuity with the previous one."
        })
    
    # Add the draft chapter content
    revision_context.append({
        "role": "user",
        "content": f"Here is the draft of Chapter {chapter_number} that needs revision:\n\n{chapter_content}\n\nPlease revise this chapter."
    })
    
    # Create the revision prompt
    revision_prompt = f"""
    Please thoroughly revise and improve this chapter with the following goals:
    
    1. OUTLINE CONFORMITY: Ensure the chapter strictly follows the outline elements for Chapter {chapter_number}
    2. SEED ADHERENCE: Verify the chapter correctly implements the original story vision from the seed
    3. CHARACTER BEHAVIOR:
       - Characters must act according to their established profiles
       - Focus on behaviors relevant to this specific moment in the story arc
       - Ensure character motivations are clear and consistent with their background
       - Dialogue should reflect each character's unique voice and perspective
    4. WRITING STYLE:
       - Adhere strictly to the period-appropriate language and setting-consistent vocabulary
       - Eliminate ANY anachronistic terms, concepts, or phrasing
       - Maintain the appropriate genre conventions and stylistic elements
       - Balance description, dialogue, and action according to seed preferences
    5. LANGUAGE VARIETY:
       - Identify and fix repetitive sentence structures, words, or patterns
       - Vary paragraph and sentence length to improve rhythm and flow
       - Ensure diverse descriptive techniques (sensory details, metaphors, etc.)
       - Reduce overuse of character names/pronouns at sentence beginnings
       - Replace generic descriptions with more specific, vivid, setting-appropriate ones
    6. HOOK INTEGRATION: Confirm all relevant hooks are properly established, developed, or resolved
    7. CONTINUITY: Maintain perfect continuity with previous/surrounding story elements
    8. SETTING REALISM: Ensure the world and setting details are consistent and realistic
    9. PROSE QUALITY: Improve writing quality, flow, dialogue, and descriptions where needed
    
    SPECIFIC REVISION PRIORITIES:
    - Find and eliminate ANY anachronistic vocabulary, idioms, or concepts
    - Identify paragraphs with repetitive sentence structures and rewrite for variety
    - Replace generic descriptions that could apply to any story with setting-specific ones
    - Enhance passages with too much "tell" and not enough "show"
    - Fix any instances where the text doesn't reflect the true nature of the time period/setting
    
    FORMAT REQUIREMENTS:
    - Maintain the chapter heading format: "## Chapter {chapter_number}: {chapter_info['title']}"
    - Ensure proper paragraph breaks for readability
    - Preserve scene breaks (if any) marked with "* * *"
    - End with a satisfying conclusion that leads naturally to the next chapter
    
    Return the complete revised chapter, making substantial improvements while maintaining the core structure and formatting.
    """
    
    revision_context.append({"role": "user", "content": revision_prompt})
    
    try:
        # Start time for progress tracking
        start_time = time.time()
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            temperature=temperature,
            messages=revision_context,
            thinking={"type": "enabled", "budget_tokens": 12000}
        )
        
        # Calculate and display generation time
        elapsed_time = time.time() - start_time
        print(f" done! ({elapsed_time:.1f} seconds)")
        
        # Extract the text content from the message using our helper function
        content_text = extract_message_content(response)
            
        # If thinking is in the response, analyze it (optional)
        thinking = None
        for content in response.content:
            if hasattr(content, 'type') and content.type == "thinking":
                if hasattr(content, 'text'):
                    thinking = content.text
                
        if thinking:
            thinking_lines = thinking.split('\n')
            thinking_sample = '\n'.join(thinking_lines[:3]) + "..." if len(thinking_lines) > 3 else thinking
            print(f"Revision thinking sample: \n{thinking_sample}")
            print(f"Used {len(thinking.split())} words in revision thinking process")
            
        return content_text
    except Exception as e:
        print(f"\nError revising chapter: {e}")
        # Return original content if revision fails
        return chapter_content

def generate_chapter(
    client: anthropic.Anthropic,
    outline: str,
    chapter_info: Dict[str, str],
    chapter_number: int,
    hooks_text: Optional[str] = None,
    profiles_text: Optional[str] = None,
    seed_text: Optional[str] = None,
    previous_chapter: Optional[str] = None,
    chapter_summaries: Optional[Dict[int, str]] = None,
    temperature: float = 1.0
) -> str:
    """Generate a single chapter using Claude with thinking capability."""
    
    # Prepare context for Claude
    chapter_context = []
    
    # Create setting and period-appropriate style guide based on seed
    if seed_text:
        # First, add the original seed
        chapter_context.append({
            "role": "user",
            "content": f"Here is the original seed concept for the story:\n\n{seed_text}\n\nThis represents the initial vision for the story."
        })
        
        # Then generate a style guide based on the setting
        style_guide_prompt = """
        Based on the seed document above, create a brief style guide that ensures period-appropriate and setting-consistent writing:
        
        1. Identify the time period, culture, and setting of the story
        2. List 5-7 linguistic patterns appropriate to this setting (vocabulary, sentence structure, etc.)
        3. List 10-15 words/terms that would be appropriate for this setting
        4. List 10-15 modern words/phrases/concepts to AVOID in this setting
        5. Identify 3-5 distinctive stylistic elements to maintain throughout
        
        This style guide will help maintain authentic, period-appropriate language and avoid anachronisms.
        """
        
        chapter_context.append({
            "role": "user",
            "content": style_guide_prompt
        })
        
        # Add a response from Claude with the style guide
        chapter_context.append({
            "role": "assistant",
            "content": "I'll create a style guide based on the seed document to ensure period-appropriate writing:"
        })
    
    # Add the outline for context
    chapter_context.append({
        "role": "user", 
        "content": f"Here is the detailed outline for the story:\n\n{outline}\n\nPlease reference this outline throughout our conversation."
    })
    
    # Add character profiles if available
    if profiles_text:
        chapter_context.append({
            "role": "user",
            "content": f"Here are detailed character profiles for the story:\n\n{profiles_text}\n\nUse these profiles to maintain consistent characterization."
        })
    
    # Add hooks if available
    if hooks_text:
        # Extract chapter-specific hooks
        chapter_hooks = extract_chapter_hooks(hooks_text, chapter_number)
        
        if 'all' in chapter_hooks:
            # If no chapter-specific hooks found, provide all hooks
            hook_content = f"Here are the strategic hooks for the story that ensure reader engagement:\n\n{chapter_hooks['all']}\n\nWeave these hooks into your chapter where appropriate."
        else:
            # Format chapter-specific hooks
            hook_content = "Here are the strategic hooks relevant to this part of the story:\n\n"
            
            if chapter_number - 1 in chapter_hooks:
                hook_content += f"Hooks from previous chapter (Chapter {chapter_number-1}):\n{chapter_hooks[chapter_number-1]}\n\n"
            
            if chapter_number in chapter_hooks:
                hook_content += f"Hooks for this chapter (Chapter {chapter_number}):\n{chapter_hooks[chapter_number]}\n\n"
            
            if chapter_number + 1 in chapter_hooks:
                hook_content += f"Hooks for upcoming chapter (Chapter {chapter_number+1}):\n{chapter_hooks[chapter_number+1]}\n\n"
            
            hook_content += "Be sure to establish, develop, or resolve these hooks as appropriate for this chapter."
        
        chapter_context.append({
            "role": "user",
            "content": hook_content
        })
    
    # Add previous chapter summaries if available
    if chapter_summaries and len(chapter_summaries) > 0:
        summaries_content = "Here are summaries of the previous chapters to maintain continuity:\n\n"
        
        # Get all previous chapter summaries up to the current chapter
        for ch_num in sorted(chapter_summaries.keys()):
            if ch_num < chapter_number:
                summaries_content += f"Chapter {ch_num} Summary:\n{chapter_summaries[ch_num]}\n\n"
        
        chapter_context.append({
            "role": "user",
            "content": summaries_content
        })
    
    # Add previous chapter if available
    if previous_chapter:
        chapter_context.append({
            "role": "user",
            "content": f"Here is the previous chapter that you wrote:\n\n{previous_chapter}\n\nPlease ensure continuity with this chapter."
        })
    
    # Determine if we need a special approach for auto-divided outlines
    # Check if the chapter description contains section headers or is a portion of the overall outline
    has_section_headers = "###" in chapter_info['description']
    
    # Request to write the chapter
    if has_section_headers:
        writing_prompt = f"""
        I need you to write Chapter {chapter_number}: {chapter_info['title']}
        
        Chapter outline:
        {chapter_info['description']}
        
        Please write a complete, engaging chapter based on this part of the outline. Focus on the sections marked with "###" within this content. Use your thinking capability to:
        1. Plan the chapter structure and key scenes based on the section headers
        2. Develop characterization with the following principles:
           - Characters must act according to their established profiles
           - Focus on behaviors relevant to this specific moment in the story arc
           - Ensure character motivations are clear and consistent with their background
           - Dialogue should reflect each character's unique voice and perspective
        3. Follow the writing style guidance from the seed document and style guide:
           - Strictly adhere to period-appropriate language and setting-consistent vocabulary
           - Avoid any anachronistic terms, concepts, or phrasing
           - Maintain the specified tone, style, and genre conventions
           - Balance description, dialogue, and action as appropriate
        4. Create vivid, varied descriptions and natural dialogue:
           - Use different sentence structures and patterns to avoid repetition
           - Vary paragraph lengths for better rhythm and pacing
           - Employ diverse descriptive techniques (sensory details, metaphors, etc.)
           - Avoid repeating the same words in close proximity
        5. Ensure proper pacing and meaningful development
        6. Integrate the strategic hooks properly (setups, developments, resolutions)
        7. Revise for clarity, coherence, variety, and impact
        
        IMPORTANT: If this chapter contains any of these common issues, actively work to avoid them:
        - Overuse of character names or pronouns at the start of sentences
        - Repetitive sentence structures or patterns
        - Anachronistic vocabulary or concepts for the time period
        - Generic descriptions that could apply to any story
        - Modern idioms or expressions in historical settings
        - Predictable scene transitions
        
        Remember that context is king! Reference all available context including the outline, hooks, character profiles, style guide, previous chapter summaries, and the full previous chapter to create a cohesive narrative that builds on what came before.
        
        Format the chapter using this structure:
        1. Start with a clear chapter heading: "## Chapter {chapter_number}: {chapter_info['title']}"
        2. Include paragraph breaks for readability
        3. You may add scene breaks with "* * *" between major scene changes if appropriate
        4. End the chapter with a satisfying conclusion that leads into the next chapter
        """
    else:
        writing_prompt = f"""
        I need you to write Chapter {chapter_number}: {chapter_info['title']}
        
        Chapter outline:
        {chapter_info['description']}
        
        Please write a complete, engaging chapter following this outline. Use your thinking capability to:
        1. Plan the chapter structure and key scenes
        2. Develop characterization with the following principles:
           - Characters must act according to their established profiles
           - Focus on behaviors relevant to this specific moment in the story arc
           - Ensure character motivations are clear and consistent with their background
           - Dialogue should reflect each character's unique voice and perspective
        3. Follow the writing style guidance from the seed document and style guide:
           - Strictly adhere to period-appropriate language and setting-consistent vocabulary
           - Avoid any anachronistic terms, concepts, or phrasing
           - Maintain the specified tone, style, and genre conventions
           - Balance description, dialogue, and action as appropriate
        4. Create vivid, varied descriptions and natural dialogue:
           - Use different sentence structures and patterns to avoid repetition
           - Vary paragraph lengths for better rhythm and pacing
           - Employ diverse descriptive techniques (sensory details, metaphors, etc.)
           - Avoid repeating the same words in close proximity
        5. Ensure proper pacing and meaningful development
        6. Integrate the strategic hooks properly (setups, developments, resolutions)
        7. Revise for clarity, coherence, variety, and impact
        
        IMPORTANT: If this chapter contains any of these common issues, actively work to avoid them:
        - Overuse of character names or pronouns at the start of sentences
        - Repetitive sentence structures or patterns
        - Anachronistic vocabulary or concepts for the time period
        - Generic descriptions that could apply to any story
        - Modern idioms or expressions in historical settings
        - Predictable scene transitions
        
        Remember that context is king! Reference all available context including the outline, hooks, character profiles, style guide, previous chapter summaries, and the full previous chapter to create a cohesive narrative that builds on what came before.
        
        Format the chapter using this structure:
        1. Start with a clear chapter heading: "## Chapter {chapter_number}: {chapter_info['title']}"
        2. Include paragraph breaks for readability
        3. You may add scene breaks with "* * *" between major scene changes if appropriate
        4. End the chapter with a satisfying conclusion that leads into the next chapter
        """
    
    chapter_context.append({"role": "user", "content": writing_prompt})
    
    # Generate the chapter with Claude
    print(f"Generating Chapter {chapter_number}: {chapter_info['title']}...")
    print("Thinking and writing in progress...", end="", flush=True)
    
    try:
        # Start time for progress tracking
        start_time = time.time()
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            temperature=temperature,
            messages=chapter_context,
            thinking={"type": "enabled", "budget_tokens": 16000}
        )
        
        # Calculate and display generation time
        elapsed_time = time.time() - start_time
        print(f" done! ({elapsed_time:.1f} seconds)")
        
        # Extract the text content from the message using our helper function
        content_text = extract_message_content(response)
            
        # If thinking is in the response, analyze it (optional)
        thinking = None
        for content in response.content:
            if hasattr(content, 'type') and content.type == "thinking":
                if hasattr(content, 'text'):
                    thinking = content.text
                
        if thinking:
            thinking_lines = thinking.split('\n')
            thinking_sample = '\n'.join(thinking_lines[:3]) + "..." if len(thinking_lines) > 3 else thinking
            print(f"Thinking sample: \n{thinking_sample}")
            print(f"Used {len(thinking.split())} words in thinking process")
            
        return content_text
    except Exception as e:
        print(f"\nError generating chapter: {e}")
        return ""

def main():
    # Load environment variables (.env file)
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Initialize Anthropic client
    client = anthropic.Anthropic()
    
    # Find and validate story files (outline, hooks, profiles, seed)
    try:
        # Check if either --story-dir or --outline is provided
        if not args.story_dir and not args.outline:
            print("Error: Either --story-dir or --outline must be provided")
            print("Use --story-dir to specify a directory containing all story files")
            print("Or use --outline to specify an outline file path (legacy mode)")
            sys.exit(1)
            
        # Find story files
        outline_path, hooks_path, profiles_path, seed_path = find_story_files(
            story_dir=args.story_dir,
            outline_path=args.outline,
            input_dir=args.input_dir,
            hooks_path=args.hooks,
            profiles_path=args.profiles,
            seed_path=args.seed
        )
        
        if not Path(outline_path).exists():
            print(f"Error: Outline file '{outline_path}' not found")
            sys.exit(1)
        
        outline_text = read_file(outline_path)
        
        # Read hooks, profiles, and seed if available
        hooks_text = read_file(hooks_path) if hooks_path else None
        profiles_text = read_file(profiles_path) if profiles_path else None
        seed_text = read_file(seed_path) if seed_path else None
        
        # Print status of found files
        print(f"Using outline: {outline_path}")
        if hooks_text:
            print(f"Using hooks: {hooks_path}")
        else:
            print("No hooks file found or specified")
        if profiles_text:
            print(f"Using profiles: {profiles_path}")
        else:
            print("No profiles file found or specified")
        if seed_text:
            print(f"Using seed: {seed_path}")
        else:
            print("No seed file found or specified")
            
    except Exception as e:
        print(f"Error preparing story files: {e}")
        sys.exit(1)
    
    # Parse the outline
    chapters = parse_outline(outline_text, target_chapters=args.target_chapters)
    
    if args.chapters:
        chapters = chapters[:args.chapters]
    
    print(f"Found {len(chapters)} chapters in the outline (format: {chapters[0]['title'] if chapters else 'unknown'})")
    
    # Generate story title from the outline
    story_title = ""
    for line in outline_text.split("\n"):
        if line.startswith("# "):
            story_title = line.lstrip("# ").strip()
            break
    
    # Set up output file and directory
    if args.output:
        output_path = Path(args.output)
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Determine the output location based on input
        if args.story_dir:
            # If using story_dir, create output in the chapters directory
            story_dir = Path(args.story_dir)
            story_name = story_dir.name
            chapters_dir = story_dir / "chapters"
            chapters_dir.mkdir(exist_ok=True)
            output_path = chapters_dir / f"{story_name}.md"
        else:
            # Legacy mode: use outline file name
            outline_filename = Path(outline_path).stem
            
            # Strip timestamp if present (e.g., england_20250311_015444 -> england)
            base_name = outline_filename.split('_')[0] if '_' in outline_filename else outline_filename
            
            # Create a dedicated folder for this story
            story_dir = Path("stories") / base_name
            story_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output file path
            output_path = story_dir / f"{base_name}.md"
    
    print(f"Story output will be saved to: {output_path}")
    
    # Check if we're starting from a specific chapter
    start_chapter = args.start_chapter
    
    # If starting from chapter > 1, we need to read previous chapters for context
    previous_chapter = None
    if start_chapter > 1 and output_path.exists():
        try:
            existing_story = read_file(str(output_path))
            
            # Simple parsing to extract the last chapter
            chapters_in_file = existing_story.split("## Chapter ")
            if len(chapters_in_file) >= start_chapter:
                # Get the last chapter before our starting point
                previous_chapter = "## Chapter " + chapters_in_file[start_chapter-1]
                print(f"Found previous chapter in existing file for continuity")
            else:
                print(f"Warning: Starting from chapter {start_chapter} but fewer chapters exist in file")
                
        except Exception as e:
            print(f"Warning: Could not read existing story file for continuity: {e}")
    
    # If starting from chapter 1 or creating a new story, initialize the file
    if start_chapter == 1 or not output_path.exists():
        # Write story title and introduction to output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {story_title}\n\n")
            f.write(f"*Generated story based on outline: {Path(outline_path).name}*\n\n")
            
            # Add seed information if available
            if seed_text:
                seed_file = Path(seed_path).name if seed_path else "Unknown"
                f.write(f"*Original seed: {seed_file}*\n\n")
                
            # Add a table of contents placeholder
            f.write("## Table of Contents\n\n")
            f.write("*[Generated upon completion]*\n\n")
            
            # Add a divider before the story content begins
            f.write("---\n\n")
    
    # Prepare chapters to generate
    chapters_to_generate = chapters[start_chapter-1:] if start_chapter <= len(chapters) else []
    
    if not chapters_to_generate:
        print(f"No chapters to generate starting from chapter {start_chapter}")
        sys.exit(0)
    
    # Generate chapters sequentially
    total_generation_time = 0
    total_revision_time = 0
    total_word_count = 0
    
    # Dictionary to store chapter summaries
    chapter_summaries = {}
    
    if args.no_revision:
        print(f"\nStarting story generation with temperature {args.temperature}")
        print(f"Using Claude thinking capability with 16000 tokens budget (revision disabled)")
    else:
        print(f"\nStarting story generation with temperature {args.temperature}, revision temperature {args.revision_temperature}")
        print(f"Using Claude thinking capability with 16000 tokens budget for generation and 12000 for revision")
    
    print(f"Generating chapters {start_chapter} to {start_chapter + len(chapters_to_generate) - 1}")
    print("-" * 60)
    
    # Create directories for chapter summaries
    if args.story_dir:
        # Use the story_dir if provided
        story_dir = Path(args.story_dir)
    else:
        # Otherwise use output_path.parent
        story_dir = output_path.parent
        
    chapters_dir = story_dir / "chapters"
    chapters_dir.mkdir(exist_ok=True)
    summaries_dir = chapters_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    
    # Check if there are existing summaries from previous runs
    # If continuing from a previous chapter, load existing summaries first
    if start_chapter > 1:
        for ch_num in range(1, start_chapter):
            # Try to load from new location first, then fall back to old location if needed
            summary_path = summaries_dir / f"chapter_{ch_num:02d}_summary.md"
            old_summary_path = story_dir / "summaries" / f"chapter_{ch_num:02d}_summary.md"
            
            # For very old stories, also check if summaries are in the root of the story directory
            root_summary_path = story_dir / f"chapter_{ch_num:02d}_summary.md"
            
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    chapter_summaries[ch_num] = f.read()
                print(f"Loaded existing summary for Chapter {ch_num}")
            elif old_summary_path.exists():
                # For backward compatibility with existing stories
                with open(old_summary_path, "r", encoding="utf-8") as f:
                    chapter_summaries[ch_num] = f.read()
                print(f"Loaded existing summary for Chapter {ch_num} from old location")
            elif root_summary_path.exists():
                # For very old stories with summaries in the root
                with open(root_summary_path, "r", encoding="utf-8") as f:
                    chapter_summaries[ch_num] = f.read()
                print(f"Loaded existing summary for Chapter {ch_num} from root directory")
    
    for i, chapter_info in enumerate(chapters_to_generate, start_chapter):
        # Track time for both generation and revision
        start_time = time.time()
        
        print(f"\n[{i}/{len(chapters)}] ", end="")
        
        # Step 1: Generate initial chapter draft
        # Note: previous_chapter will be the revised version of the previous chapter if available
        chapter_content = generate_chapter(
            client=client,
            outline=outline_text,
            chapter_info=chapter_info,
            chapter_number=i,
            hooks_text=hooks_text,
            profiles_text=profiles_text,
            seed_text=seed_text,
            previous_chapter=previous_chapter,  # This is already the revised version from previous iteration
            chapter_summaries=chapter_summaries,  # Pass all previous chapter summaries
            temperature=args.temperature
        )
        
        if not chapter_content:
            print(f"Failed to generate Chapter {i}")
            continue
        
        # Calculate chapter generation time
        generation_time = time.time() - start_time
        
        # If revision is enabled, proceed with revision step
        if not args.no_revision:
            # Step 2: Revise the chapter
            revision_start_time = time.time()
            revised_chapter = revise_chapter(
                client=client,
                chapter_content=chapter_content,
                outline=outline_text,
                chapter_info=chapter_info,
                chapter_number=i,
                hooks_text=hooks_text,
                profiles_text=profiles_text,
                seed_text=seed_text,
                previous_chapter=previous_chapter,  # Use the revised version from previous iteration
                temperature=args.revision_temperature
            )
            
            # Calculate revision time
            revision_time = time.time() - revision_start_time
            total_revision_time += revision_time
            
            # Store revised chapter for context in the next generation
            previous_chapter = revised_chapter  # This revised chapter will be used in the next iteration
            chapter_to_save = revised_chapter
            
            # Step 3: Generate bullet point summary of the revised chapter
            chapter_summary = generate_chapter_summary(
                client=client,
                chapter_content=revised_chapter,
                chapter_number=i,
                chapter_title=chapter_info['title']
            )
            
            # Store the chapter summary for future chapters
            chapter_summaries[i] = chapter_summary
            
            # Save the summary to its own file
            summary_path = summaries_dir / f"chapter_{i:02d}_summary.md"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(chapter_summary)
            print(f"  Chapter summary saved to: {summary_path}")
            
            # Calculate total chapter time
            chapter_time = time.time() - start_time
            
            # Word count
            word_count = len(revised_chapter.split())
            
            print(f"Completed Chapter {i} of {len(chapters)}")
            print(f"  Words: {word_count} | Total Time: {chapter_time:.1f}s (Gen: {generation_time:.1f}s, Rev: {revision_time:.1f}s)")
            print(f"  Speed: {word_count/chapter_time:.1f} words/sec")
        else:
            # Skip revision, use the initial chapter
            previous_chapter = chapter_content  # Store for next iteration
            chapter_to_save = chapter_content
            
            # For non-revised chapters, still generate a summary
            chapter_summary = generate_chapter_summary(
                client=client,
                chapter_content=chapter_content,
                chapter_number=i,
                chapter_title=chapter_info['title']
            )
            
            # Store the chapter summary for future chapters
            chapter_summaries[i] = chapter_summary
            
            # Save the summary to its own file
            summary_path = summaries_dir / f"chapter_{i:02d}_summary.md"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(chapter_summary)
            print(f"  Chapter summary saved to: {summary_path}")
            
            # Calculate total chapter time
            chapter_time = time.time() - start_time
            
            # Word count
            word_count = len(chapter_content.split())
            
            print(f"Completed Chapter {i} of {len(chapters)}")
            print(f"  Words: {word_count} | Time: {chapter_time:.1f}s | Speed: {word_count/chapter_time:.1f} words/sec")
            
        # Append chapter to the story file (use append mode since we might be continuing)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"{chapter_to_save}\n\n")
            
        # Write individual chapter file in chapters subfolder
        if args.story_dir:
            # Use the story_dir if provided
            story_dir = Path(args.story_dir)
        else:
            # Otherwise use output_path.parent
            story_dir = output_path.parent
        
        chapters_dir = story_dir / "chapters"
        chapters_dir.mkdir(exist_ok=True)
        chapter_filename = f"chapter_{i:02d}.md"
        chapter_path = chapters_dir / chapter_filename
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(f"{chapter_to_save}")
        print(f"  Individual chapter saved to: {chapter_path}")
        
        total_generation_time += generation_time
        total_word_count += word_count
        
        print("-" * 60)
        
        # Slight pause between API calls to avoid rate limits
        if i < len(chapters):
            time.sleep(1)
    
    chapters_generated = len(chapters_to_generate)
    avg_generation_time = total_generation_time / chapters_generated if chapters_generated else 0
    avg_words_per_chapter = total_word_count / chapters_generated if chapters_generated else 0
    
    print(f"\nStory generation complete!")
    print(f"  Chapters generated: {chapters_generated} (from {start_chapter} to {start_chapter + chapters_generated - 1})")
    print(f"  Total chapters in story: {len(chapters)}")
    print(f"  Total words generated: {total_word_count}")
    print(f"  Avg words per chapter: {avg_words_per_chapter:.1f}")
    
    if args.no_revision:
        print(f"  Total generation time: {total_generation_time:.1f} seconds")
        print(f"  Avg generation time per chapter: {avg_generation_time:.1f} seconds")
        print(f"  Overall speed: {total_word_count/total_generation_time:.1f} words/sec")
    else:
        avg_revision_time = total_revision_time / chapters_generated if chapters_generated else 0
        avg_total_time = (total_generation_time + total_revision_time) / chapters_generated if chapters_generated else 0
        
        print(f"  Total generation time: {total_generation_time:.1f} seconds")
        print(f"  Total revision time: {total_revision_time:.1f} seconds")
        print(f"  Total processing time: {total_generation_time + total_revision_time:.1f} seconds")
        print(f"  Avg generation time per chapter: {avg_generation_time:.1f} seconds")
        print(f"  Avg revision time per chapter: {avg_revision_time:.1f} seconds")
        print(f"  Avg total time per chapter: {avg_total_time:.1f} seconds")
        print(f"  Overall speed: {total_word_count/(total_generation_time + total_revision_time):.1f} words/sec")
    
    print(f"  Story saved to: {output_path}")
    
    # Generate and update the table of contents if we've completed at least one chapter
    if chapters_generated > 0:
        print("Generating table of contents...")
        try:
            # Read the current content
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Generate TOC
            toc = generate_table_of_contents(content)
            
            # Replace the placeholder TOC with the actual TOC
            if "## Table of Contents" in content:
                # Split the content to find where to insert the TOC
                parts = content.split("## Table of Contents")
                
                if len(parts) >= 2:
                    # Find the end of the TOC section
                    toc_end = parts[1].find("---")
                    if toc_end != -1:
                        # Recombine with the new TOC
                        new_content = parts[0] + toc + "\n\n---" + parts[1][toc_end + 3:]
                        
                        # Write the updated content
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        
                        print("Table of contents updated successfully.")
        except Exception as e:
            print(f"Error updating table of contents: {e}")

if __name__ == "__main__":
    main()