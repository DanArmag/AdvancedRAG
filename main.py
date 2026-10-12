import tkinter as tk

from ui.app import ChatApp


def main():
    root = tk.Tk()

    app = ChatApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()