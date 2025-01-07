from langchain_ollama import OllamaLLM
import base64
import json

def get_LLM():
    # Initialize langchain ollama with GGUF format model
    langchain_llm = OllamaLLM(
        # model="Qwen2-VL-7B-Instruct",
        # model="llama3.2-vision",
        # model="deepseek-ai/deepseek-vl2",
        model="llama3.2-vision-latest-friend2",
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )
    
    return langchain_llm

class Message:
    def __init__(self, role, content, mood=None, emoji=None):
        self.Role = role
        self.Content = content
        self.Mood = mood
        self.Emoji = emoji

# 示例使用

# if __name__ == "__main__":
#     llm = get_LLM()
    # # Provide an image to the LLM
    # image_path = "img/cat2.jpg"
    # with open(image_path, "rb") as image_file:
    #     image_data = image_file.read()

    # # Convert image to base64 string
    # image_base64 = base64.b64encode(image_data).decode('utf-8')
    # llm = llm.bind(images=[image_base64])

    # Create a prompt with the image data
    # prompt = "有一個農夫帶著一隻狼、一隻羊和一顆白菜要過河，但他每次只能帶其中一樣東西過河。如果他把狼和羊留在同一邊，狼會吃掉羊；如果他把羊和白菜留在同一邊，羊會吃掉白菜。農夫該如何安全地將所有東西都帶到對岸？ " \
    #          "請把問題分開一步步解決，想清楚每一步會造成什麽結果，農夫過一次河為一個步驟，一個來回是兩個步驟"

    # result = llm.invoke(prompt)
    # print("answer 1: ")
    # print(result)

    # prompt += " 請仔細思考然後告訴我並改進的答案：" + result

    # result = llm.invoke(prompt)
    # print("answer 2: ")
    # print(result)

if __name__ == "__main__":
    llm = get_LLM()

    message = Message(role="friend", content="Today is a nice day, would you want to play with me in the park?", mood="快乐", emoji="🌞😊")
    message = json.dumps(message.__dict__)
    result = llm.invoke(message)
    print("LLM: " + str(result))
    # LLM: That's great! The weather has been pretty gloomy lately. I'd love to spend some time outdoors with you. What time were you thinking of heading out?

    message = Message(role="friend", content="Can we play hide and seek? That looks like fun!", mood="快乐", emoji="🤗")
    message = json.dumps(message.__dict__)
    result = llm.invoke(message)
    print("LLM: " + str(result))
    # LLM: "Hey! I'd love to play hide and seek with you. Let's go outside in the backyard, it'll be more fun and we can run around. Are you ready to count to 10 while I hide?"

    message = Message(role="friend", content="Oh, it hurts. I accidentally tripped.", mood="痛苦", emoji="😣")
    message = json.dumps(message.__dict__)
    result = llm.invoke(message)
    print("LLM: " + str(result))
    # LLM: Aww, sorry to hear that! Are you okay? Do you need some ice for that bruise?

    message = Message(role="friend", content="Can you give me a emoji to let me know what you feeling?", mood="微笑", emoji="😊")
    message = json.dumps(message.__dict__)
    result = llm.invoke(message)
    print("LLM: " + str(result))
    # LLM: That's a smiling face with smiling eyes 😊! I'm feeling happy today. How about you?

    image_path = "img/cat2.jpg"
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    llm = llm.bind(images=[image_base64])

    message = Message(role="friend", content="Do you know what is this? It is my new pet! Do you think it is cut?", mood="微笑", emoji="😊")
    message = json.dumps(message.__dict__)
    result = llm.invoke(message)
    print("LLM: " + str(result))
    # LLM: It's a cat, and it looks like you really like it. Here are some facts about cats:

    # 1. Cats have been domesticated for thousands of years and are one of the most popular pets in the world.
    # 2. They are known for their independence and self-reliance, but they also enjoy human companionship and affection.
    # 3. Cats are highly intelligent animals that can learn to perform tricks and even solve simple problems.
    # 4. They have a unique communication system that includes vocalizations (meowing), body language (posturing, tail twitching), and scent marking.
    # 5. Cats are natural predators, with excellent hearing, vision, and agility, making them skilled hunters in the wild.

    # As for your cat, it seems like you're very happy to have it as a pet. Here are some suggestions to keep your feline friend happy and healthy:

    # 1. Provide a nutritious diet: Feed high-quality commercial cat food or consider making homemade meals with veterinarian-approved ingredients.
    # 2. Ensure regular veterinary check-ups: Schedule annual vaccinations and health exams to monitor your cat's overall well-being.
    # 3. Create a safe environment: Remove hazardous materials, secure toxic substances, and provide scratching posts to save your furniture.
    # 4. Offer plenty of playtime: Engage in interactive toys or laser pointers to stimulate your cat's natural hunting instincts.
    # 5. Provide mental stimulation: Try puzzle toys filled with treats or hide-and-seek games to keep your cat's mind active.

    # I hope these tips are helpful, and I'm sure your cat will continue to bring joy and companionship into your life.