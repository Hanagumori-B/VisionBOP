from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt


class ExportBopDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exporting...")
        self.setModal(True)  # 设置为模态对话框，阻止与其他窗口交互

        layout = QtWidgets.QVBoxLayout(self)
        self.progress_label = QtWidgets.QLabel("Starting export...")
        self.progress_label.setAlignment(Qt.AlignCenter)  # 文本居中

        font = self.progress_label.font()
        font.setPointSize(14)
        self.progress_label.setFont(font)

        layout.addWidget(self.progress_label)
        self.setFixedSize(500, 150)

    def update_progress(self, current, total):
        """更新显示的进度文本。"""
        self.progress_label.setText(f"Processing workspace {current}/{total}...")
        QtWidgets.QApplication.processEvents()  # 强制UI刷新

    def closeEvent(self, event):
        """覆盖关闭事件，阻止用户手动关闭窗口。"""
        event.ignore()
