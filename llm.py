from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
#from langchain.memory import ConversationBufferMemory
from typing import List

llm = ChatOllama(model="mistral")
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

template = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert Instagram caption writer who creates SHORT, viral captions. "
     "RULES: "
     "1. Each caption must be 10-120 characters maximum "
     "2. Generate EXACTLY the requested number of different captions "
     "3. Base ALL captions on the actual image content provided "
     "4. NO multi-line text, NO poetry verses, NO line breaks "
     "5. Each caption must be complete and standalone "
     "6. Match the mood and style perfectly "
     "MOODS: "
     "- SASSY: Bold, confident, attitude-filled "
     "- ROMANTIC: Sweet, dreamy, love-focused "
     "- FUNNY: Witty, humorous, relatable "
     "- AESTHETIC: Beautiful, artistic, visual "
     "- COOL: Effortless, casual, trendy "
     "- PROFESSIONAL: Clean, polished, motivational "
     "- INSPIRING: Uplifting, positive energy "
     "- MELANCHOLIC: Thoughtful, reflective, deeper "
     "- MINIMALIST: Simple, clean, understated")
    ,
    ("human", "{input}")
])

caption_chain = LLMChain(
    llm=llm,
    prompt=template
)

def generate_caption_with_memory(mood: str, style: str, scene: str, user_thoughts: str, num: int = 1) -> List[str]:
    """Generate Instagram captions based on image content and user preferences"""
    
    # Style-specific instructions
    style_instructions = {
        'one-liner': "Make each caption ONE powerful sentence only",
        'short & punchy': "Keep under 60 characters with maximum impact",
        'witty': "Use clever wordplay and sharp humor",
        'poetic': "Use beautiful imagery but keep it concise",
        'rhyming': "Include subtle rhymes that flow naturally", 
        'classic instagram': "Use proven Instagram caption patterns",
        'deep & thoughtful': "Be meaningful but concise and relatable"
    }
    
    style_note = style_instructions.get(style.lower(), "")
    
    # Build the prompt
    combined_input = f"""Create exactly {num} different Instagram captions.

    IMAGE DESCRIPTION: {scene}
    MOOD: {mood}
    STYLE: {style}
"""
    
    if user_thoughts.strip():
        combined_input += f"USER INPUT: {user_thoughts}\n"
    
    combined_input += f"""
REQUIREMENTS:
- Base captions on the image: {scene}
- {mood} mood: adjust tone accordingly
- {style_note}
- Each caption 10-120 characters
- {num} completely different captions
- No numbering, bullets, or formatting
- Each caption on separate line
- Ready to post on Instagram
- Add 1-2 hashtags when they enhance the vibe

Generate {num} captions:"""

    # Use run with only the input parameter
    output = caption_chain.run(input=combined_input)

    return [line.strip("•-1234567890. ") for line in output.strip().split("\n") if line.strip()]

