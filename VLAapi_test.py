#!/usr/bin/env python3
"""
快速测试脚本 - 电脑屏幕理解VLM测试
用于测试计算机使用agent的VLA模型（屏幕描述、UI元素识别、GUI操作推理）
"""

import openai
import base64
from PIL import Image
import io
import sys
import os

# 配置
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

def encode_image(image_path):
    """将图像文件编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_image_if_needed(image_path, max_size=1024):
    """如果图像太大，进行缩放（可选）"""
    try:
        img = Image.open(image_path)
        if max(img.size) > max_size:
            # 等比例缩放
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 保存到临时文件
            temp_path = "/tmp/test_resized.png"
            img.save(temp_path)
            return temp_path
    except Exception as e:
        print(f"⚠️ 图像处理失败: {e}")
    
    return image_path

def test_text_only():
    """纯文本测试"""
    print("\n" + "="*50)
    print("📝 测试1: 纯文本对话")
    print("="*50)
    
    try:
        models = client.models.list()
        model_id = models.data[0].id
        print(f"✅ 使用模型: {model_id}")
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": "你好，请用一句话介绍你自己"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print("\n🤖 回复:")
        print(response.choices[0].message.content)
        
        if response.usage:
            print(f"\n📊 Token使用: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文本测试失败: {e}")
        return False

def test_vlm_image_description(image_path="test.png"):
    """VLM测试：电脑屏幕描述"""
    print("\n" + "="*50)
    print("🖥️ 测试2: VLM - 电脑屏幕描述")
    print("="*50)

    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        print("请确保test.png在当前目录")
        return False

    try:
        # 如果需要，调整图像大小
        processed_path = resize_image_if_needed(image_path)

        # 编码图像
        base64_image = encode_image(processed_path)

        models = client.models.list()
        model_id = models.data[0].id

        print(f"📸 分析屏幕截图: {image_path}")

        # 构建消息 - Qwen2-VL/UI-TARS 格式
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "这是一张电脑屏幕截图。请详细描述屏幕上显示的内容，包括：1. 窗口和应用程序；2. 用户界面元素（按钮、菜单、图标等）；3. 当前显示的信息或内容；4. 任何可见的文本内容。"
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.7
        )

        print("\n🤖 屏幕描述:")
        print(response.choices[0].message.content)

        if response.usage:
            print(f"\n📊 Token使用: {response.usage.total_tokens}")

        # 清理临时文件
        if processed_path != image_path and os.path.exists(processed_path):
            os.remove(processed_path)

        return True

    except Exception as e:
        print(f"❌ VLM测试失败: {e}")
        print("\n💡 提示: 如果失败，可能是因为:")
        print("  1. 模型不支持图像输入")
        print("  2. API格式不兼容")
        print("  3. 图像太大")
        return False

def test_vlm_question_about_image(image_path="test.png"):
    """VLM测试：屏幕操作相关问答"""
    print("\n" + "="*50)
    print("❓ 测试3: VLM - 屏幕操作问答")
    print("="*50)

    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return False

    try:
        base64_image = encode_image(image_path)

        models = client.models.list()
        model_id = models.data[0].id

        # 预设与电脑屏幕操作相关的问题
        questions = [
            "屏幕当前显示的是什么应用程序或界面？这个界面有什么主要功能和元素？",
            "屏幕上的哪些元素可以作为点击按钮（可交互的UI元素）？请列出所有可点击的按钮、链接或图标。",
            "如果要在这个屏幕上执行一个操作（如打开文件、搜索、提交表单等），应该点击哪个具体位置？请描述坐标位置或元素名称。",
            "屏幕显示的内容是否存在错误提示或警告信息？如果有，请详细描述。"
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n📌 问题{i}: {question[:50]}...")

            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )

            print(f"🤖 回答: {response.choices[0].message.content}")

        return True

    except Exception as e:
        print(f"❌ VLM问答测试失败: {e}")
        return False

def test_vlm_multiple_images():
    """测试多图输入（如果模型支持）- 用于GUI自动化场景"""
    print("\n" + "="*50)
    print("🖥️🖥️ 测试4: VLM - 多屏幕状态对比（可选）")
    print("="*50)

    # 检查是否有多个测试图像
    test_images = ["test1.png", "test2.png"]  # 可以修改为你实际的文件名

    existing_images = [img for img in test_images if os.path.exists(img)]

    if len(existing_images) < 2:
        print("⚠️ 需要至少2张屏幕截图进行多图测试，跳过")
        print("   这对于测试GUI自动化中的状态变化检测很有用")
        return False

    try:
        base64_images = [encode_image(img) for img in existing_images]

        models = client.models.list()
        model_id = models.data[0].id

        # 构建多图消息 - 针对GUI操作场景
        content = []
        for i, base64_img in enumerate(base64_images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}"
                }
            })

        content.append({
            "type": "text",
            "text": "这是两帧电脑屏幕截图。请比较它们：1. 界面发生了什么变化？2. 是否有新的窗口打开或关闭？3. 是否有按钮被点击或状态改变？4. 用户可能执行了什么操作导致这些变化？"
        })

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=300,
            temperature=0.7
        )

        print("\n🤖 屏幕状态变化分析:")
        print(response.choices[0].message.content)

        return True

    except Exception as e:
        print(f"❌ 多图测试失败: {e}")
        return False

def create_sample_image():
    """如果没有测试图像，创建一个模拟电脑屏幕的示例图像"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # 创建一个模拟电脑桌面的图像
        img = Image.new('RGB', (800, 600), color='#1e1e1e')  # 深色桌面背景
        draw = ImageDraw.Draw(img)

        # 绘制任务栏
        draw.rectangle([0, 550, 800, 600], fill='#333333')
        # 任务栏左侧图标
        draw.rectangle([10, 555, 40, 590], fill='#4a90d9')
        draw.text((15, 560), "⊞", fill='white')
        draw.rectangle([50, 555, 80, 590], fill='#4a90d9')
        draw.text((55, 560), "□", fill='white')
        # 任务栏右侧时间
        draw.text((700, 560), "12:30", fill='white')

        # 绘制一个窗口
        draw.rectangle([100, 80, 500, 500], fill='#252526', outline='#3c3c3c', width=2)
        # 窗口标题栏
        draw.rectangle([100, 80, 500, 110], fill='#333333')
        draw.text((110, 90), "文件管理器", fill='white')
        # 窗口关闭按钮
        draw.rectangle([470, 85, 495, 105], fill='#e81123')
        draw.text((478, 90), "×", fill='white')

        # 窗口内容区 - 侧边栏
        draw.rectangle([105, 115, 200, 495], fill='#252526')
        draw.text((115, 130), "快速访问", fill='#cccccc')
        draw.text((115, 160), "📁 桌面", fill='white')
        draw.text((115, 190), "📁 文档", fill='white')
        draw.text((115, 220), "📁 下载", fill='white')

        # 窗口内容区 - 主区域
        draw.rectangle([205, 115, 495, 495], fill='#1e1e1e')
        # 模拟文件/文件夹
        draw.text((220, 130), "📁 项目文件夹", fill='white')
        draw.text((220, 170), "📄 readme.txt", fill='white')
        draw.text((220, 210), "📄 report.pdf", fill='white')
        draw.text((220, 250), "🖼️ screenshot.png", fill='white')

        # 底部状态栏
        draw.rectangle([100, 470, 500, 500], fill='#007acc')
        draw.text((110, 480), "3 个项目", fill='white')

        # 绘制另一个应用窗口（记事本）
        draw.rectangle([520, 150, 780, 450], fill='#ffffff', outline='#000000', width=1)
        draw.rectangle([520, 150, 780, 175], fill='#ececec')
        draw.text((530, 160), "记事本 - untitled.txt", fill='black')
        draw.rectangle([765, 155, 775, 170], fill='#e81123')
        draw.text((535, 200), "这是示例文本内容。", fill='black')
        draw.text((535, 230), "用于测试屏幕截图理解能力。", fill='black')

        img.save("test.png")
        print("✅ 已创建模拟电脑屏幕截图: test.png")
        return True
    except Exception as e:
        print(f"❌ 创建示例图像失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("🤖 电脑屏幕理解 VLM测试脚本")
    print("   用于评估VLA模型的GUI操作能力")
    print("="*60)
    
    # 检查服务是否可用
    try:
        models = client.models.list()
        print(f"✅ 服务连接成功，可用模型: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"❌ 无法连接到vLLM服务: {e}")
        print("请确保服务已启动: vllm serve ...")
        return
    
    # 检查test.png
    if not os.path.exists("test.png"):
        print("\n⚠️ 未找到test.png")
        choice = input("是否创建示例图像用于测试? (y/n): ")
        if choice.lower() == 'y':
            create_sample_image()
        else:
            print("请准备test.png后重试")
            return
    
    # 运行测试
    tests = [
        ("文本测试", test_text_only),
        ("图像描述测试", lambda: test_vlm_image_description("test.png")),
        ("图像问答测试", lambda: test_vlm_question_about_image("test.png")),
        ("多图测试", test_vlm_multiple_images)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🚀 运行: {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except KeyboardInterrupt:
            print("\n⚠️ 测试被中断")
            break
        except Exception as e:
            print(f"❌ 测试出错: {e}")
            results.append((test_name, False))
    
    # 显示测试结果汇总
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {test_name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！模型正常运行。")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
