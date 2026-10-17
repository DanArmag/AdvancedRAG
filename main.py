
import tkinter as tk

from ui.app import ChatApp


def main():
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
