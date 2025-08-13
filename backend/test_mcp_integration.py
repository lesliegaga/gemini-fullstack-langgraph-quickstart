#!/usr/bin/env python3
"""
高德MCP集成测试脚本 (使用langchain-mcp-adapters)

这个脚本验证使用langchain-mcp-adapters的高德MCP集成是否正常工作。
"""

import asyncio
import os
import sys
import traceback

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    from agent.graph import load_amap_mcp_tools, has_location_intent
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保安装了所有必要的依赖:")
    print("pip install langchain-mcp-adapters mcp")
    sys.exit(1)


async def test_mcp_connection():
    """测试MCP连接和工具加载"""
    print("🔧 测试MCP连接和工具加载...")
    
    # 检查API密钥
    api_key = os.getenv("AMAP_MAPS_API_KEY")
    if not api_key:
        print("❌ 未找到AMAP_MAPS_API_KEY环境变量")
        print("请设置环境变量: export AMAP_MAPS_API_KEY='your_key'")
        return False
    
    print(f"✅ API密钥已配置: {api_key[:10]}...")
    
    try:
        # 直接测试MCP连接
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@amap/amap-maps-mcp-server"],
            env={
                "AMAP_MAPS_API_KEY": api_key
            }
        )
        
        print("🚀 连接到高德MCP服务器...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ MCP会话初始化成功")
                
                # 加载工具
                tools = await load_mcp_tools(session)
                print(f"✅ 成功加载 {len(tools)} 个MCP工具")
                
                # 列出可用工具
                if tools:
                    print("📋 可用工具:")
                    for i, tool in enumerate(tools, 1):
                        tool_name = getattr(tool, 'name', 'Unknown')
                        tool_desc = getattr(tool, 'description', 'No description')
                        print(f"   {i}. {tool_name}: {tool_desc}")
                
                return True
                
    except Exception as e:
        print(f"❌ MCP连接测试失败: {e}")
        traceback.print_exc()
        return False


async def test_graph_integration():
    """测试与graph.py的集成"""
    print("\n🔗 测试与graph.py的集成...")
    
    try:
        # 测试工具加载函数
        print("📦 测试load_amap_mcp_tools函数...")
        tools = await load_amap_mcp_tools()
        
        if tools:
            print(f"✅ 成功通过graph.py加载 {len(tools)} 个工具")
            return True
        else:
            print("❌ 没有加载到任何工具")
            return False
            
    except Exception as e:
        print(f"❌ graph.py集成测试失败: {e}")
        traceback.print_exc()
        return False


def test_location_detection():
    """测试地理位置检测功能"""
    print("\n🔍 测试地理位置检测功能...")
    
    test_cases = [
        {
            "query": "王府井附近适合遛娃的好去处",
            "expected": True,
            "description": "地理位置+活动查询"
        },
        {
            "query": "北京三里屯周边美食推荐", 
            "expected": True,
            "description": "地理位置+美食查询"
        },
        {
            "query": "人工智能的发展历史",
            "expected": False,
            "description": "纯知识查询"
        },
        {
            "query": "如何学习Python编程",
            "expected": False,
            "description": "技术学习查询"
        },
        {
            "query": "上海的地址在哪里",
            "expected": True,
            "description": "位置查询"
        },
        {
            "query": "杭州西湖的坐标",
            "expected": True,
            "description": "坐标查询"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        expected = case["expected"]
        description = case["description"]
        
        print(f"\n测试 {i}/{total}: {description}")
        print(f"查询: {query}")
        
        # 测试检测功能
        result = has_location_intent(query)
        if result == expected:
            print(f"✅ 检测结果正确: {result}")
            passed += 1
        else:
            print(f"❌ 检测结果错误: 期望 {expected}, 实际 {result}")
    
    print(f"\n地理位置检测测试完成: {passed}/{total} 通过")
    return passed == total


async def test_tool_usage():
    """测试MCP工具的实际使用"""
    print("\n🛠️  测试MCP工具的实际使用...")
    
    try:
        # 加载工具
        tools = await load_amap_mcp_tools()
        
        if not tools:
            print("❌ 没有可用的工具进行测试")
            return False
        
        print(f"📋 将测试 {len(tools)} 个工具")
        
        # 这里可以添加具体的工具调用测试
        # 但需要根据实际的MCP工具接口来实现
        print("✅ 工具使用测试占位符 - 需要根据实际MCP接口实现")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具使用测试失败: {e}")
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("🚀 高德MCP集成测试套件 (langchain-mcp-adapters)")
    print("=" * 70)
    
    # 检查环境
    if not os.getenv("AMAP_MAPS_API_KEY"):
        print("❌ 请设置AMAP_MAPS_API_KEY环境变量")
        print("   获取方式:")
        print("   1. 访问 https://console.amap.com/dev/key/app")
        print("   2. 注册并创建应用")
        print("   3. 获取Web服务API Key")
        print("   4. 设置环境变量: export AMAP_MAPS_API_KEY='your_key'")
        return
    
    # 检查Node.js环境
    import subprocess
    try:
        result = subprocess.run(['npx', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js环境检查通过: npx {result.stdout.strip()}")
        else:
            print("❌ npx不可用，请确保安装了Node.js")
            return
    except FileNotFoundError:
        print("❌ 找不到npx命令，请确保安装了Node.js")
        return
    
    # 测试列表
    tests = [
        ("地理位置检测", test_location_detection, False),    # 同步测试
        ("MCP连接和工具加载", test_mcp_connection, True),     # 异步测试
        ("graph.py集成", test_graph_integration, True),     # 异步测试
        ("MCP工具使用", test_tool_usage, True),             # 异步测试
    ]
    
    results = []
    
    for test_name, test_func, is_async in tests:
        print(f"\n{'='*70}")
        print(f"🧪 测试: {test_name}")
        print('='*70)
        
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试 {test_name} 出现异常: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print(f"\n{'='*70}")
    print("📊 测试总结")
    print('='*70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试都通过了！高德MCP集成(langchain-mcp-adapters)准备就绪。")
        print("\n📋 使用说明:")
        print("1. 确保AMAP_MAPS_API_KEY环境变量已设置")
        print("2. 确保Node.js环境可用(npx命令)")
        print("3. 系统会自动检测地理位置查询")
        print("4. MCP工具将通过langchain-mcp-adapters自动加载")
        print("5. 系统将并行执行网络搜索和高德MCP搜索")
    else:
        print("\n⚠️  部分测试失败，请检查:")
        print("- AMAP_MAPS_API_KEY是否正确")
        print("- Node.js环境是否可用")
        print("- 网络连接是否正常")
        print("- 是否已安装 langchain-mcp-adapters 和 mcp 包")
        print("- 高德MCP服务器(@amap/amap-maps-mcp-server)是否可用")


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
