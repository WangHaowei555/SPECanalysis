import tkinter as tk
from DataAnalysis import *
from datadraw import *

class mygui():
    def __init__(self):
        self.window = tk.Tk()
        self.title = self.window.title('SPECanalysis')
        self.geometry = self.window.geometry('1440x1440')

    #主菜单
    def main_menu(self):
        menubar = tk.Menu(self.window)
        menubar.add_command(label='文件')

    def maintain(self):
        self.window.config(menu=self.main_menu)
        self.window.mainloop()


if __name__ == '__main__':
    New_window = mygui()
    New_window.main_menu()
    New_window.maintain()