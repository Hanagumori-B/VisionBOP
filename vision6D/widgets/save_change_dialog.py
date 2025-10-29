from PyQt5 import QtWidgets


class SaveChangesDialog(QtWidgets.QDialog):
    """A custom dialog to ask the user if they want to save changes."""
    Save = 1
    Discard = 2
    Cancel = 3

    def __init__(self, parent=None, workspace='', handle_name='Switch'):
        super().__init__(parent)
        self.setWindowTitle(f"Unsaved Changes: {workspace}")

        layout = QtWidgets.QVBoxLayout(self)

        message = QtWidgets.QLabel(
            f"The current workspace {workspace} has been modified. Do you want to save the changes before {handle_name}?")
        layout.addWidget(message)

        # Create button box with custom buttons
        button_box = QtWidgets.QDialogButtonBox(self)
        self.save_button = button_box.addButton(f"Save & {handle_name}", QtWidgets.QDialogButtonBox.AcceptRole)
        self.dont_save_button = button_box.addButton(f"Don't Save & {handle_name}", QtWidgets.QDialogButtonBox.DestructiveRole)
        self.cancel_button = button_box.addButton("Cancel", QtWidgets.QDialogButtonBox.RejectRole)

        # Connect signals to custom slots
        self.save_button.clicked.connect(self.on_save)
        self.dont_save_button.clicked.connect(self.on_discard)
        self.cancel_button.clicked.connect(self.on_cancel)

        layout.addWidget(button_box)

        self.result = None

    def on_save(self):
        self.result = self.Save
        self.accept()  # Closes the dialog

    def on_discard(self):
        self.result = self.Discard
        self.accept()  # Closes the dialog

    def on_cancel(self):
        self.result = self.Cancel
        self.reject()  # Closes the dialog

    def exec_(self):
        super().exec_()
        return self.result
