import json


class BopJsonEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        if 'indent' not in kwargs:
            kwargs['indent'] = 0  # 默认内层缩进为4
        super().__init__(*args, **kwargs)
        self._is_toplevel = True  # 增加一个状态标记，用于识别顶层对象

    def iterencode(self, obj, _one_shot=False):
        # 检查是否为顶层字典
        if self._is_toplevel and isinstance(obj, dict):
            self._is_toplevel = False  # 进入后立即重置标记，防止递归时误判

            yield "{\n"  # 手动添加开头的花括号和换行符

            # 获取字典的最后一个键，用于判断是否添加末尾的逗号
            last_key = None
            if obj:
                last_key = list(obj.keys())[-1]

            # 遍历字典的键值对
            for key, value in obj.items():
                # 产出键
                if isinstance(self.indent, int):
                    yield self.indent * " "  # 添加一级的缩进
                yield json.dumps(key, ensure_ascii=self.ensure_ascii)
                yield ": "

                # 递归地对值进行编码，此时 _is_toplevel 已经是 False
                # 所以内层的对象会使用父类的标准格式化方法
                yield from super().iterencode(value)

                # 如果不是最后一个键值对，则添加逗号和换行符
                if key != last_key:
                    yield ",\n"

            yield "\n}"  # 手动添加结尾的换行符和花括号

            # 完成后恢复标记，以便同一个encoder实例可以被复用
            self._is_toplevel = True
        else:
            # 如果不是顶层字典，或者不是字典类型，则使用父类的默认行为
            yield from super().iterencode(obj)
