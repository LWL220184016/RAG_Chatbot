from langchain_ollama import OllamaLLM
import base64

def get_LLM():
    # Initialize langchain ollama with GGUF format model
    langchain_llm = OllamaLLM(
        # model="Qwen2-VL-7B-Instruct",
        # model="llama3.2-vision",
        model="deepseek-ai/deepseek-vl2",
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )
    
    return langchain_llm

def process_output(output):
    # Process the output from the generator
    print("Processing output:", output)

if __name__ == "__main__":
    llm = get_LLM()
    # # Provide an image to the LLM
    # image_path = "img/cat2.jpg"
    # with open(image_path, "rb") as image_file:
    #     image_data = image_file.read()

    # # Convert image to base64 string
    # image_base64 = base64.b64encode(image_data).decode('utf-8')
    # llm = llm.bind(images=[image_base64])

    # Create a prompt with the image data
    prompt = "有一個農夫帶著一隻狼、一隻羊和一顆白菜要過河，但他每次只能帶其中一樣東西過河。如果他把狼和羊留在同一邊，狼會吃掉羊；如果他把羊和白菜留在同一邊，羊會吃掉白菜。農夫該如何安全地將所有東西都帶到對岸？ " \
             "請把問題分開一步步解決，想清楚每一步會造成什麽結果，農夫過一次河為一個步驟，一個來回是兩個步驟"

    result = llm.invoke(prompt)
    print("answer 1: ")
    print(result)

    prompt += " 請仔細思考然後告訴我並改進的答案：" + result

    result = llm.invoke(prompt)
    print("answer 2: ")
    print(result)