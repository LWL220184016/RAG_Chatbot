import base64

image_path = "img/cat2.jpg"
with open(image_path, "rb") as image_file:
    image_data = image_file.read()

# Convert image to base64 string
image_base64 = base64.b64encode(image_data).decode('utf-8')
print(image_base64)
