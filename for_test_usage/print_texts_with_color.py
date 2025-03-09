text = "Hello, World!"

# # 注解中的顔色有部分是錯誤的
# print(f"\n\033[38;5;196m{text}\033[0m")  # 红色高亮 (256-color)
# print(f"\n\033[38;5;208m{text}\033[0m")  # 橙色高亮 (256-color)
# print(f"\n\033[38;5;202m{text}\033[0m")  # 深橙色高亮 (256-color)
# print(f"\n\033[38;5;226m{text}\033[0m")  # 黄色高亮 (256-color)

# print(f"\n\033[38;5;01m{text}01m\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;02m{text}02m\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;03m{text}03m\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;04m{text}04m\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;05m{text}05m\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;06m{text}06m\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;07m{text}07m\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;08m{text}08m\033[0m")  # 绿色高亮 (256-color)

# print(f"\n\033[38;5;03m{text}\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;04m{text}\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;05m{text}\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;10m{text}\033[0m")  # 绿色高亮 (256-color)
# print(f"\n\033[38;5;11m{text}\033[0m")  # 青色高亮 (256-color)
# print(f"\n\033[38;5;12m{text}\033[0m")  # 蓝色高亮 (256-color)
# print(f"\n\033[38;5;13m{text}\033[0m")  # 紫色高亮 (256-color)
# print(f"\n\033[38;5;14m{text}\033[0m")  # 粉色高亮 (256-color)
# print(f"\n\033[38;5;15m{text}\033[0m")  # 白色高亮 (256-color)
# print(f"\n\033[38;5;16m{text}\033[0m")  # 黑色高亮 (256-color)
# print(f"\n\033[38;5;17m{text}\033[0m")  # 灰色高亮 (256-color)
# print(f"\n\033[38;5;18m{text}\033[0m")  # 橙色高亮 (256-color)
# print(f"\n\033[38;5;19m{text}\033[0m")  # 黄色高亮 (256-color)

for i in range(256):
    print(f"\n\033[38;5;{i}m{text}{i}m\033[0m")