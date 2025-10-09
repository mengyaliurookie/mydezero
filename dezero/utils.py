import os
import subprocess
from dezero import Variable


def _dot_var(v, verbose=False):
    # 确保标签使用UTF-8编码兼容的格式，转义特殊字符
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        # 处理可能导致乱码的特殊字符
        name += f"{v.shape} {v.dtype}"

    # 使用双引号包裹标签，避免单引号冲突
    return f'"{id(v)}" [label="{name}", color=orange, style=filled]\n'


def _dot_func(f):
    txt = f'"{id(f)}" [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    for x in f.inputs:
        txt += f'"{id(x)}" -> "{id(f)}"\n'
    for y in f.outputs:
        y_var = y()
        if y_var is not None:
            txt += f'"{id(f)}" -> "{id(y_var)}"\n'
    return txt


def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    if output.creator is not None:
        add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # 1. 保存DOT文件（指定UTF-8编码避免乱码）
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    os.makedirs(tmp_dir, exist_ok=True)  # 简化目录创建
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    # 关键：用UTF-8编码写入，避免中文等字符乱码
    with open(graph_path, 'w', encoding='utf-8') as f:
        f.write(dot_graph)

    # 2. 调用dot命令（修复命令执行问题）
    extension = os.path.splitext(to_file)[1][1:]
    # 处理未指定扩展名的情况
    if not extension:
        extension = 'png'
        to_file = f'{to_file}.{extension}'

    try:
        # 关键：使用列表形式传递参数，禁用shell，捕获输出
        result = subprocess.run(
            ['dot', graph_path, '-T', extension, '-o', to_file],
            check=True,
            shell=False,  # 禁用shell可避免路径解析问题
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        print(f"成功生成图形：{os.path.abspath(to_file)}")
    except FileNotFoundError:
        raise RuntimeError(
            "未找到dot命令！请先安装Graphviz并添加到系统PATH\n"
            "下载地址：https://graphviz.org/download/"
        )
    except subprocess.CalledProcessError as e:
        # 输出详细错误信息用于调试
        print(f"Graphviz错误输出：{e.stderr}")
        print(f"生成的DOT内容：\n{dot_graph}")
        raise RuntimeError(f"生成图形失败，错误代码：{e.returncode}")
