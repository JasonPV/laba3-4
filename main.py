import sys
import matplotlib.pyplot as plt
from methods import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        lbl_get_best = QLabel('1) Get best params:', self)
        lbl_get_best.move(1, 1)

        lbl_get_examples = QLabel('2) Get examples:', self)
        lbl_get_examples.move(1, 350)

        lbl_get_cross = QLabel('3) Get cross validate:', self)
        lbl_get_cross.move(1, 650)

        lbl_sart_text = QLabel('4) Voting method:', self)
        lbl_sart_text.move(1200, 1)
        ###########################################################
        #split data
        lbl_split = QLabel(self)
        lbl_split.setText('Enter the number of the test')
        self.qle_1 = QLineEdit(self)
        self.qle_1.move(100, 100)
        lbl_split.move(60, 40)
        btn_split = QPushButton('Ok', self)
        btn_split.clicked.connect(self.on_click_split)
        btn_split.move(120, 160)

        #choose method
        lbl_method = QLabel('Choose method', self)
        self.combo_method = QComboBox(self)
        self.combo_method.addItems(["histogram", "dft", "dct", "gradient", "scale"])
        lbl_method.move(500, 40)
        self.combo_method.move(507, 100)
        btn_method = QPushButton('Choose', self)
        btn_method.clicked.connect(self.on_click_method)
        btn_method.move(510, 160)

        #best param and score
        lbl_score_on_test = QLabel('Score on test:', self)
        lbl_best_param = QLabel('Best param:', self)
        self.lbl_1 = QLabel('?', self)
        self.lbl_2 = QLabel('?', self)
        lbl_score_on_test.move(800, 40)
        self.lbl_1.move(800, 70)
        lbl_best_param.move(800, 120)
        self.lbl_2.move(800, 150)
        #########################################################
        #choose method
        lbl_method_2 = QLabel('Choose method' ,self)
        self.combo_method_2 = QComboBox(self)
        self.combo_method_2.addItems(["histogram", "dft", "dct", "gradient", "scale"])
        btn_method_2 = QPushButton('Choose', self)
        btn_method_2.clicked.connect(self.on_click_method_2)
        lbl_method_2.move(100, 400)
        self.combo_method_2.move(100, 460)
        btn_method_2.move(100, 520)

        #enter param
        lbl_param = QLabel('Enter parametr', self)
        self.qle_param = QLineEdit(self)
        btn_param = QPushButton('Ok', self)
        btn_param.clicked.connect(self.on_click_param)
        lbl_param.move(500, 400)
        self.qle_param.move(500, 460)
        btn_param.move(500, 520)
        self.lbl_pixmap_1 = QLabel(self)
        self.lbl_pixmap_2 = QLabel(self)
        self.lbl_pixmap_3 = QLabel(self)
        self.lbl_pixmap_1.move(700, 400)
        self.lbl_pixmap_2.move(850, 400)
        self.lbl_pixmap_3.move(1000, 400)
        self.lbl_plot_1 = QLabel(self)
        self.lbl_plot_2 = QLabel(self)
        self.lbl_plot_3 = QLabel(self)
        self.lbl_plot_1.move(700, 530)
        self.lbl_plot_2.move(850, 530)
        self.lbl_plot_3.move(1000, 530)

        ########################################################

        #choose method
        lbl_method_3 = QLabel('Choose method', self)
        self.combo_method_3 = QComboBox(self)
        self.combo_method_3.addItems(["histogram", "dft", "dct", "gradient", "scale"])
        btn_method_3 = QPushButton('Choose', self)
        btn_method_3.clicked.connect(self.on_click_method_3)
        lbl_method_3.move(100, 700)
        self.combo_method_3.move(100, 760)
        btn_method_3.move(100, 820)

        self.lbl_pixmap_cross = QLabel(self)
        self.lbl_pixmap_cross.move(300, 700)

        self.lbl_best_p = QLabel(self)
        self.lbl_a = QLabel(self)
        self.lbl_n = QLabel(self)
        self.lbl_best_p1 = QLabel('Best param:', self)
        self.lbl_a1 = QLabel('Accuracy:', self)
        self.lbl_n1 = QLabel('Number of train:', self)
        self.lbl_best_p1.move(650, 700)
        self.lbl_a1.move(750, 700)
        self.lbl_n1.move(850, 700)
        self.lbl_best_p.move(650, 750)
        self.lbl_a.move(750, 750)
        self.lbl_n.move(850, 750)

        ##################################################
        #start
        btn_start = QPushButton('Start', self)
        btn_start.move(1250, 40)
        btn_start.clicked.connect(self.on_click_start)
        self.lbl_plot = QLabel(self)
        self.lbl_plot.move(1250, 250)

        self.lbl_best_par = QLabel(self)
        self.lbl_ac = QLabel(self)
        self.lbl_num = QLabel(self)
        self.lbl_ac1 = QLabel('Accuracy:', self)
        self.lbl_num1 = QLabel('Number of train:', self)
        self.lbl_ac1.move(1400, 40)
        self.lbl_num1.move(1500, 40)
        self.lbl_num.move(1500, 90)
        self.lbl_ac.move(1400, 90)

        self.table = QTableWidget(self)
        self.table.setColumnCount(7)
        self.table.setRowCount(10)
        col = 0
        for i in ['train', 'get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale', 'voting']:
            cellinfo = QTableWidgetItem(i)
            self.table.setItem(0, col, cellinfo)
            col += 1
        row = 1
        for i in range(10):
            cellinfo = QTableWidgetItem(str(i+1))
            self.table.setItem(row, 0, cellinfo)
            row += 1
        self.table.move(1650, 40)

        ########################################################
        self.setGeometry(QDesktopWidget().availableGeometry())
        self.setWindowTitle('Classifier')
        self.show()

    def on_click_split(self):
        self.test = int(self.qle_1.text())

    def on_click_method(self):
        data = get_data()
        data_train, data_test = get_split_data(data, self.test)
        best_p, accuracy = get_best_params(data_train, data_test, eval('get_'+self.combo_method.currentText()))
        self.lbl_1.setText(str(accuracy))
        self.lbl_1.adjustSize()
        self.lbl_2.setText(str(best_p))
        self.lbl_2.adjustSize()

    def on_click_method_2(self):
        self.method_2 = eval('get_'+self.combo_method_2.currentText())

    def on_click_param(self):
        data = get_data()
        samples, r = get_samples(data)
        samples = samples[0]
        images = random.choices(samples, k=3)
        for i in range(3):
            image = cv2.resize(images[i], (90, 90), interpolation=cv2.INTER_AREA)
            cv2.imwrite('test.jpg', 255*image)
            pixmap = QPixmap('test.jpg')
            eval(f'self.lbl_pixmap_{i+1}').setPixmap(pixmap)
            eval(f'self.lbl_pixmap_{i+1}').adjustSize()
            if self.method_2 == get_histogram:
                hist, bins = get_histogram(images[i], int(self.qle_param.text()))
                hist = np.insert(hist, 0, 0.0)
                fig = plt.figure(figsize=(1.1, 1.1))
                ax = fig.add_subplot(111)
                ax.plot(bins, hist)
                plt.xticks(color='w')
                plt.yticks(color='w')
                plt.savefig('plot.png')
                plt.close(fig)
            elif self.method_2 == get_dct or self.method_2 == get_dft:
                ex = self.method_2(images[i], int(self.qle_param.text()))
                fig = plt.figure(figsize=(1.1, 1.1))
                ax = fig.add_subplot(111)
                ax.pcolormesh(range(ex.shape[0]), range(ex.shape[0]), np.flip(ex, 0))
                plt.xticks(color='w')
                plt.yticks(color='w')
                plt.savefig('plot.png')
                plt.close(fig)
            elif self.method_2 == get_scale:
                image = self.method_2(images[i], float(self.qle_param.text()))
                cv2.imwrite('plot.png', 255 * image)
            else:
                ex = self.method_2(images[i], int(self.qle_param.text()))
                fig = plt.figure(figsize=(1.1, 1.1))
                ax = fig.add_subplot(111)
                ax.plot(range(0, len(ex)), ex)
                plt.xticks(color='w')
                plt.yticks(color='w')
                plt.savefig('plot.png')
                plt.close(fig)

            pixmap1 = QPixmap('plot.png')
            eval(f'self.lbl_plot_{i+1}').setPixmap(pixmap1)
            eval(f'self.lbl_plot_{i+1}').adjustSize()

    def on_click_method_3(self):
        data = get_data()
        method = eval('get_'+self.combo_method_3.currentText())
        ans, ps = get_cross(data, method)
        fig, ax = plt.subplots(figsize=(3.2, 2.5))
        # plt.figure(figsize=(2.5, 2.5))
        # fig = plt.figure(figsize=(2.5, 2.5))
        # ax = fig.add_subplot(111)
        plt.plot(range(1, len(ans)+1), ans)
        plt.grid(True)
        plt.xlabel("Number of test")
        plt.ylabel("Accuracy")
        plt.savefig('plot1.png')
        # plt.show()
        plt.close()
        pixmap2 = QPixmap('plot1.png')
        self.lbl_pixmap_cross.setPixmap(pixmap2)
        self.lbl_pixmap_cross.adjustSize()
        self.lbl_best_p.setText(str(round(ps[ans.index(max(ans))], 2)))
        self.lbl_best_p.adjustSize()
        self.lbl_n.setText(str(ans.index(max(ans))+1))
        self.lbl_n.adjustSize()
        self.lbl_a.setText(str(max(ans)))
        self.lbl_a.adjustSize()

    def on_click_start(self):
        data = get_data()
        accuracy = []
        ps = []
        for i in range(9, 0, -1):
            data_train, data_test = get_split_data(data, i)
            r, p = get_vote(data_train, data_test)
            a = accuracy_score(r, data_test[1])
            p.append(a)
            accuracy.append(a)
            ps.append(p)

        fig, ax = plt.subplots(figsize=(3.2, 2.5))
        plt.plot(range(1, len(accuracy) + 1), accuracy)
        plt.grid(True)
        plt.xlabel("Number of test")
        plt.ylabel("Accuracy")
        plt.savefig('plot1.png')
        plt.close()
        pixmap2 = QPixmap('plot1.png')
        self.lbl_plot.setPixmap(pixmap2)
        self.lbl_plot.adjustSize()
        self.lbl_ac.setText(str(max(accuracy)))
        self.lbl_ac.adjustSize()
        self.lbl_num.setText(str(accuracy.index(max(accuracy))+1))
        self.lbl_num.adjustSize()


        for i in range(len(ps)):
            for j in range(6):
                cellinfo = QTableWidgetItem(str(ps[i][j]))
                self.table.setItem(i+1, j+1, cellinfo)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())