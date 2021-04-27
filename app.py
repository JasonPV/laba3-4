import sys
import matplotlib.pyplot as plt
import collections
from statistics import mean
from methods import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class mainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        #tab 1
        btn_example = QPushButton('get example', self)
        btn_num = QPushButton('get image', self)

        layout_images = QHBoxLayout(spacing=50)
        layout_features = QHBoxLayout(spacing=50)
        self.lbls = []
        for i in range(4):
            self.lbls.append(QLabel(self))

        for i in range(2):
            layout_images.addWidget(self.lbls[i])
            layout_features.addWidget(self.lbls[i+2])

        layout_images.addStretch(1)
        layout_features.addStretch(1)

        btn_example.clicked.connect(self.on_click_example)
        btn_num.clicked.connect(self.on_click_num_example)
        self.qle_split = QLineEdit()
        self.qle_number_of_image = QLineEdit()
        # layout_num_image = QHBoxLayout(spacing=50)
        # layout_num_image.addWidget(btn_num)
        # layout_num_image.addWidget(self.qle_split)
        # layout_num_image.addWidget(self.qle_number_of_image)
        # layout_num_image.addStretch(1)

        lbl_param_1 = QLabel('Choose param:', self)
        lbl_method_1 = QLabel('Choose method:', self)
        self.combo_method_1 = QComboBox(self)
        self.combo_method_1.addItems(["histogram", "dft", "dct", "gradient", "scale"])

        self.tab_1 = QFrame()
        self.layout_tab_1 = QVBoxLayout(self, spacing=50)
        self.layout_part = QHBoxLayout(spacing=50)
        self.layout_full = QHBoxLayout(spacing=50)
        self.layout_line = QHBoxLayout(spacing=50)
        self.layout_second_line = QHBoxLayout(spacing=50)
        self.lbl_image = []
        self.lbl_feature = []
        for i in range(5):
            self.lbl_image.append(QLabel(self))
            self.lbl_feature.append(QLabel(self))
        self.layout_part.addWidget(btn_example, alignment=Qt.AlignLeft)

        self.vbox = []
        for i in range(5):
            self.vbox.append(QVBoxLayout(spacing=50))
            self.layout_part.addLayout(self.vbox[i])
            self.vbox[i].addWidget(self.lbl_image[i])
            self.vbox[i].addWidget(self.lbl_feature[i])
            self.vbox[i].addStretch(1)

        layout_txt = QHBoxLayout(spacing=130)
        layout_txt.addStretch(3)
        layout_txt.addWidget(QLabel('Split:'))
        # layout_txt.addStretch(2)
        layout_txt.addWidget(QLabel('Number of image:'))
        layout_txt.addStretch(30)

        self.qle_exapmle = QLineEdit()
        self.layout_line.addWidget(lbl_method_1, alignment=Qt.AlignLeft)
        self.layout_line.addWidget(lbl_param_1)
        self.layout_tab_1.addLayout(self.layout_line)
        self.layout_tab_1.addLayout(self.layout_second_line)
        self.layout_second_line.addWidget(self.combo_method_1)
        self.layout_second_line.addWidget(self.qle_exapmle)
        # self.layout_tab_1.addWidget(self.combo_method_1, alignment=Qt.AlignLeft)
        self.layout_tab_1.addLayout(self.layout_part)
        self.layout_tab_1.addLayout(self.layout_full)
        self.layout_part.addStretch(1)

        # self.layout_full.addWidget(layout_num_image)
        # self.layout_full.addStretch(1)
        # self.layout_tab_1.addLayout(layout_txt)
        # self.layout_tab_1.addLayout(layout_num_image)
        # self.layout_tab_1.addLayout(layout_images)
        # self.layout_tab_1.addLayout(layout_features)
        self.layout_tab_1.addStretch(1)
        self.layout_line.addStretch(1)
        self.layout_second_line.addStretch(1)
        self.tab_1.setLayout(self.layout_tab_1)
        ###########################################
        #tab2
        btn_start_best = QPushButton('start', self)
        btn_start_best.clicked.connect(self.on_click_best_param)
        lbl_method_2 = QLabel('Choose method:', self)
        lbl_param_2 = QLabel('Split data:', self)
        self.combo_method_2 = QComboBox(self)
        self.combo_method_2.addItems(["histogram", "dft", "dct", "gradient", "scale"])
        self.qle_test_2 = QLineEdit()

        layout_line_2 = QHBoxLayout(spacing=50)
        layout_line_2.addWidget(lbl_method_2)
        layout_line_2.addWidget(lbl_param_2)
        layout_line_2.addStretch(1)

        layout_line_22 = QHBoxLayout(spacing=50)
        layout_line_22.addWidget(self.combo_method_2)
        layout_line_22.addWidget(self.qle_test_2)
        layout_line_22.addStretch(1)


        layout_btn_and_par = QHBoxLayout(spacing=50)
        # layout_btn_and_par.addWidget(btn_start_best)
        layout_btn_and_par.addWidget(QLabel('Param:', self))
        layout_btn_and_par.addSpacing(47)
        self.qle_best_param = QLineEdit()
        # self.qle_best_param.setReadOnly(True)
        layout_btn_and_par.addWidget(self.qle_best_param)
        layout_btn_and_par.addWidget(QLabel('Accuracy on test:', self))
        self.qle_best_ac = QLineEdit()
        self.qle_best_ac.setReadOnly(True)
        layout_btn_and_par.addWidget(self.qle_best_ac)
        layout_btn_and_par.addStretch(1)

        layout = QHBoxLayout(spacing=50)
        layout.addWidget(QLabel('Num of image:'))
        layout.addWidget(self.qle_number_of_image)
        layout.addStretch(1)

        self.tab_2 = QFrame()
        self.layout_tab_2 = QVBoxLayout(spacing=50)
        self.layout_tab_2.addLayout(layout_line_2)
        self.layout_tab_2.addLayout(layout_line_22)
        self.layout_tab_2.addWidget(btn_start_best, alignment=Qt.AlignLeft)
        self.layout_tab_2.addLayout(layout_btn_and_par)
        self.layout_tab_2.addLayout(layout)
        self.layout_tab_2.addWidget(btn_num, alignment=Qt.AlignLeft)
        self.layout_tab_2.addLayout(layout_images)
        self.layout_tab_2.addLayout(layout_features)
        # self.layout_tab_2.addLayout(layout_boxes_2)
        self.layout_tab_2.addStretch(1)
        self.tab_2.setLayout(self.layout_tab_2)

        ###########################################
        #tab 3

        self.combo_method_3 = QComboBox(self)
        self.combo_method_3.addItems(["histogram", "dft", "dct", "gradient", "scale"])
        btn_start_cross_validate = QPushButton('start', self)
        btn_start_cross_validate.clicked.connect(self.on_click_cross_validate)

        self.qle_cv_p = []
        self.qle_cv_a = []
        for i in range(9):
            self.qle_cv_a.append(QLineEdit())
            self.qle_cv_p.append(QLineEdit())
            self.qle_cv_a[i].setReadOnly(True)
            self.qle_cv_p[i].setReadOnly(True)

        layout_cv_a = QHBoxLayout(spacing=50)
        layout_cv_b = QHBoxLayout(spacing=50)
        layout_cv_b.addWidget(QLabel('Best param:'))
        layout_cv_a.addWidget(QLabel('Accuracy:    '))
        for i in range(9):
            layout_cv_b.addWidget(self.qle_cv_p[i])
            layout_cv_a.addWidget(self.qle_cv_a[i])
        layout_cv_b.addStretch(1)
        layout_cv_a.addStretch(1)


        layout_numbers = QHBoxLayout(spacing = 167)
        # layout_numbers.addWidget(QLabel('Size of train:'))
        layout_numbers.addStretch(3)
        # layout_numbers.setContentsMargins(10, 10, 10, 10)
        for i in range(1, 10):
            layout_numbers.addWidget(QLabel(str(i)))
        layout_numbers.addStretch(5)

        self.lbl_cv_plot = QLabel(self)


        self.tab_3 = QFrame()
        self.layout_tab_3 = QVBoxLayout(spacing=50)
        self.layout_tab_3.addWidget(QLabel('Choose method:'))
        self.layout_tab_3.addWidget(self.combo_method_3, alignment=Qt.AlignLeft)
        self.layout_tab_3.addWidget(btn_start_cross_validate, alignment=Qt.AlignLeft)
        self.layout_tab_3.addLayout(layout_numbers)
        self.layout_tab_3.addLayout(layout_cv_b)
        self.layout_tab_3.addLayout(layout_cv_a)
        self.layout_tab_3.addWidget(self.lbl_cv_plot)
        self.layout_tab_3.addStretch(1)
        self.tab_3.setLayout(self.layout_tab_3)



        #################################################
        #tab 4
        self.qle_split_4 = QLineEdit()
        layout_split_data = QHBoxLayout(spacing=92)
        layout_split_data.addWidget(QLabel('Split data:'))
        layout_split_data.addWidget(self.qle_split_4)
        layout_split_data.addStretch(1)


        layout_methods = QHBoxLayout(spacing=70)
        layout_methods.addWidget(QLabel('Methods:        '))
        for i in ["histogram            ", "dft                      ", "dct                       ", "gradient               ", "scale                  ", "voting"]:
            layout_methods.addWidget(QLabel(i))
        layout_methods.addStretch(1)

        layout_accuracy = QHBoxLayout(spacing=50)
        layout_accuracy.addWidget(QLabel('Accuracy on test:'))
        layout_params = QHBoxLayout(spacing=50)
        layout_params.addWidget(QLabel('Param:                '))
        self.qle_ps = []
        self.qle_acs = []
        for i in range(6):
            self.qle_acs.append(QLineEdit())
            layout_accuracy.addWidget(self.qle_acs[i])
            self.qle_acs[i].setReadOnly(True)
            self.qle_ps.append(QLineEdit())
            layout_params.addWidget(self.qle_ps[i])
        self.qle_ps[-1].setReadOnly(True)
        self.qle_ps[-1].setText('Not param')
        layout_params.addStretch(1)
        layout_accuracy.addStretch(1)

        btn_start_4 = QPushButton('start', self)
        btn_start_4.clicked.connect(self.on_click_start_4)

        self.tab_4 = QFrame()
        self.layout_tab_4 = QVBoxLayout(spacing=50)
        self.layout_tab_4.addLayout(layout_split_data)
        self.layout_tab_4.addLayout(layout_methods)
        self.layout_tab_4.addLayout(layout_accuracy)
        self.layout_tab_4.addLayout(layout_params)
        self.layout_tab_4.addWidget(btn_start_4, alignment=Qt.AlignLeft)
        self.layout_tab_4.addStretch(1)
        self.tab_4.setLayout(self.layout_tab_4)


        ###################################################
        self.tab_5 = QFrame()
        btn_start_5 = QPushButton('start', self)
        btn_start_5.clicked.connect(self.on_click_start_5)
        layout_txt_5 = QHBoxLayout(spacing=120)
        layout_txt_5.addSpacing(70)
        for i in ["histogram", "dft    ", "dct", "gradient", "scale", "voting"]:
            layout_txt_5.addWidget(QLabel(i))
        layout_txt_5.addStretch(1)

        layout_h = QHBoxLayout(spacing=30)
        layouts = []
        layouts_b = QVBoxLayout(spacing=71)
        layouts_b.addSpacing(20)
        layout_h.addLayout(layouts_b)
        for j in range(9):
            layouts_b.addWidget(QLabel(str(j+1)))
        layouts_b.addStretch(1)

        for i in range(6):
            layouts.append(QVBoxLayout(spacing=10))
            layout_h.addLayout(layouts[i])
        self.qles = []
        for i in range(5):
            self.qles.append([])
            for j in range(18):
                self.qles[i].append(QLineEdit())
                self.qles[i][j].setReadOnly(True)
                layouts[i].addWidget(self.qles[i][j])
                if j%2 == 1:
                    layouts[i].addSpacing(20)
            layouts[i].addStretch(1)
        self.qle_a_f = []
        layouts[5].addSpacing(18)
        for i in range(9):
            self.qle_a_f.append(QLineEdit())
            self.qle_a_f[i].setReadOnly(True)
            layouts[5].addWidget(self.qle_a_f[i])
            layouts[5].addSpacing(53)
        layouts[5].addStretch(1)


        self.lbl_final = QLabel(self)
        layout_h.addWidget(self.lbl_final)
        layout_h.addStretch(1)



        self.layout_tab_5 = QVBoxLayout(spacing=50)
        self.layout_tab_5.addWidget(btn_start_5, alignment=Qt.AlignLeft)
        self.layout_tab_5.addLayout(layout_txt_5)
        self.layout_tab_5.addLayout(layout_h)

        self.layout_tab_5.addStretch(1)
        self.tab_5.setLayout(self.layout_tab_5)
        ###################################################

        ###################################################
        self.tab = QTabWidget(self)
        self.tab.addTab(self.tab_1, "Example")
        self.tab.addTab(self.tab_2, "Best param")
        self.tab.addTab(self.tab_3, "Cross-validate")
        self.tab.addTab(self.tab_4, 'Voting')
        self.tab.addTab(self.tab_5, 'Voting cross-validate')

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.tab)
        self.setLayout(main_layout)

        self.setGeometry(QDesktopWidget().availableGeometry())
        self.setWindowTitle('Classifier')
        self.show()

    def get_plot(self, method, im, best_p, ax):
        if method == get_histogram:
            hist, bins = get_histogram(im, best_p)
            hist = np.insert(hist, 0, 0.0)
            # fig = plt.figure(figsize=(1.1, 1.1))
            # ax = fig.add_subplot(111)
            ax.plot(bins, hist)
            # plt.xticks(color='w')
            # plt.yticks(color='w')
            # plt.savefig('plot.png')
            # plt.close(fig)
        elif method == get_dct or method == get_dft:
            ex = method(im, best_p)
            # fig = plt.figure(figsize=(1.1, 1.1))
            # ax = fig.add_subplot(111)
            ax.pcolormesh(range(ex.shape[0]), range(ex.shape[0]), np.flip(ex, 0), cmap='gray')
            # plt.xticks(color='w')
            # plt.yticks(color='w')
            # plt.savefig('plot.png')
            # plt.close(fig)
        elif method == get_scale:
            image = cv2.resize(im, (100, 100), interpolation=cv2.INTER_AREA)
            image = method(image, best_p)
            cv2.imwrite('plot.png', 255 * image)
            image = plt.imread('plot.png')
            ax.imshow(image, cmap='gray')

        else:
            ex = method(im, best_p)
            # fig = plt.figure(figsize=(1.1, 1.1))
            # ax = fig.add_subplot(111)
            ax.plot(range(0, len(ex)), ex)
            # plt.xticks(color='w')
            # plt.yticks(color='w')
            # plt.savefig('plot.png')
            # plt.close(fig)

    def get_plot_1(self, method, im, best_p):
        if method == get_histogram:
            hist, bins = get_histogram(im, best_p)
            hist = np.insert(hist, 0, 0.0)
            fig = plt.figure(figsize=(1.1, 1.1))
            ax = fig.add_subplot(111)
            plt.plot(bins, hist)
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.savefig('plot.png')
            plt.close(fig)
        elif method == get_dct or method == get_dft:
            ex = method(im, best_p)
            fig = plt.figure(figsize=(1.1, 1.1))
            ax = fig.add_subplot(111)
            plt.pcolormesh(range(ex.shape[0]), range(ex.shape[0]), np.flip(ex, 0), cmap='gray')
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.savefig('plot.png')
            plt.close(fig)
        elif method == get_scale:
            image = cv2.resize(im, (100, 100), interpolation=cv2.INTER_AREA)
            image = method(image, best_p)
            image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
            cv2.imwrite('plot.png', 255 * image)

        else:
            ex = method(im, best_p)
            fig = plt.figure(figsize=(1.1, 1.1))
            ax = fig.add_subplot(111)
            plt.plot(range(0, len(ex)), ex)
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.savefig('plot.png')
            plt.close(fig)

    def on_click_num_example(self):
        method = eval('get_' + self.combo_method_2.currentText())
        test = int(self.qle_test_2.text())
        if method == get_scale:
            param = float(self.qle_best_param.text())
        else:
            param = int(self.qle_best_param.text())
        num = int(self.qle_number_of_image.text())
        data = get_data()
        data_train, data_test = get_split_data(data, test)
        ind = closest(data_train, data_test[0][num], method, param)

        im = cv2.resize(data_test[0][num], (100, 100), interpolation=cv2.INTER_AREA)
        cv2.imwrite('test.jpg', 255 * im)
        pixmap = QPixmap('test.jpg')
        self.lbls[0].setPixmap(pixmap)
        self.lbls[0].adjustSize()

        im = cv2.resize(data_train[0][ind], (100, 100), interpolation=cv2.INTER_AREA)
        cv2.imwrite('test.jpg', 255 * im)
        pixmap = QPixmap('test.jpg')
        self.lbls[1].setPixmap(pixmap)
        self.lbls[1].adjustSize()

        self.get_plot_1(method, data_test[0][num], param)
        pixmap = QPixmap('plot.png')
        self.lbls[2].setPixmap(pixmap)
        self.lbls[2].adjustSize()

        self.get_plot_1(method, data_train[0][ind], param)
        pixmap = QPixmap('plot.png')
        self.lbls[3].setPixmap(pixmap)
        self.lbls[3].adjustSize()




    def on_click_example(self):
        method = eval('get_'+self.combo_method_1.currentText())
        if method == get_scale:
            param = float(self.qle_exapmle.text())
        else:
            param = int(self.qle_exapmle.text())

        data = get_data()
        for i in range(2):
            image = cv2.resize(data[0][i], (100, 100), interpolation=cv2.INTER_AREA)
            cv2.imwrite('test.jpg', 255 * image)
            pixmap = QPixmap('test.jpg')
            self.lbl_image[i].setPixmap(pixmap)
            self.lbl_image[i].adjustSize()
            self.get_plot_1(method, data[0][i], param)
            pixmap1 = QPixmap('plot.png')
            self.lbl_feature[i].setPixmap(pixmap1)
            self.lbl_feature[i].adjustSize()

        image = cv2.resize(data[0][50], (100, 100), interpolation=cv2.INTER_AREA)
        cv2.imwrite('test.jpg', 255 * image)
        pixmap = QPixmap('test.jpg')
        self.lbl_image[2].setPixmap(pixmap)
        self.lbl_image[2].adjustSize()
        self.get_plot_1(method, data[0][50], param)
        pixmap1 = QPixmap('plot.png')
        self.lbl_feature[2].setPixmap(pixmap1)
        self.lbl_feature[2].adjustSize()

        for i in range(3, 5):
            ind = random.randint(3, len(data[0])-1)
            image = cv2.resize(data[0][ind], (100, 100), interpolation=cv2.INTER_AREA)
            cv2.imwrite('test.jpg', 255 * image)
            pixmap = QPixmap('test.jpg')
            self.lbl_image[i].setPixmap(pixmap)
            self.lbl_image[i].adjustSize()
            self.get_plot_1(method, data[0][ind], param)
            pixmap1 = QPixmap('plot.png')
            self.lbl_feature[i].setPixmap(pixmap1)
            self.lbl_feature[i].adjustSize()

    def on_click_best_param(self):
        method = eval('get_' + self.combo_method_2.currentText())
        test = int(self.qle_test_2.text())
        data = get_data()
        data_train, data_test = get_split_data(data, test)
        if self.qle_best_param.text() == '':
            best_p, a = get_best_params(data_train, data_test, method)
            self.qle_best_param.setText(str(round(best_p, 2)))
            self.qle_best_ac.setText(str(round(a, 2)))
        else:
            if method == get_scale:
                best_p = float(self.qle_best_param.text())
            else:
                best_p = int(self.qle_best_param.text())

            r = classifier(data_train, data_test, method, best_p)
            a = accuracy_score(r, data_test[1])
            self.qle_best_ac.setText(str(round(a, 2)))
        # self.qle_best_param.setReadOnly(True)

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.set_title('Test imgage:')
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.set_title('Closest imgae:')
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax = fig.add_subplot(3, 1, 3)
        ax.set_xlim(0, len(data_test[0]))
        ax.set_ylim(0, 1)
        plt.ion()
        res = []

        for i in range(len(data_test[0])):
            ind = closest(data_train, data_test[0][i], method, best_p)
            im = data_train[0][ind]
            if data_test[1][i] == data_train[1][ind]:
                res.append(1)
            else:
                res.append(0)

        for i in range(len(data_test[0])):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax.clear()
            ax1.set_title('Test image:')
            ax2.set_title('Closest image:')
            ax.set_xlabel('Test number')
            ax.set_ylabel('Accuracy')
            ax.set_xlim(0, len(data_test[0]))
            ax.set_ylim(0, 1.2)
            image = cv2.resize(data_test[0][i], (100, 100), interpolation=cv2.INTER_AREA)
            cv2.imwrite('test.jpg', 255 * image)
            image = plt.imread('test.jpg')
            ax1.imshow(image, cmap='gray')

            ind = closest(data_train, data_test[0][i], method, best_p)
            im = data_train[0][ind]
            # if data_test[1][i] == data_train[1][ind]:
            #     res.append(1)
            # else:
            #     res.append(0)

            ax.plot([i for i in range(len(res))], [mean(res[:i+1]) for i in range(len(res))])

            im = cv2.resize(im, (100, 100), interpolation=cv2.INTER_AREA)
            cv2.imwrite('test.jpg', 255 * im)
            image = plt.imread('test.jpg')
            ax2.imshow(image, cmap='gray')

            self.get_plot(method, data_test[0][i], best_p, ax3)

            self.get_plot(method, im, best_p, ax4)
            plt.pause(2)
            fig.show()
            fig.canvas.draw()

    def on_click_cross_validate(self):
        data = get_data()
        method = eval('get_' + self.combo_method_3.currentText())
        ans, ps = get_cross(data, method)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        # ax = fig.add_subplot(111)
        plt.plot([i for i in range(1, len(ans) + 1)], ans)
        plt.grid(True)
        plt.title('cross validate')
        plt.xlabel("Number of train")
        plt.ylabel("Accuracy")
        plt.savefig('plot1.png')
        # plt.show()
        plt.close()
        pixmap = QPixmap('plot1.png')
        self.lbl_cv_plot.setPixmap(pixmap)
        self.lbl_cv_plot.adjustSize()
        for i in range(len(ans)):
            self.qle_cv_a[i].setText(str(round(ans[i], 2)))
            self.qle_cv_p[i].setText(str(round(ps[i], 2)))

    def on_click_start_4(self):
        test = int(self.qle_split_4.text())
        methods = ["histogram", "dft", "dct", "gradient", "scale"]
        data = get_data()
        data_train, data_test = get_split_data(data, test)
        result = []
        for i in range(len(methods)):
            method = eval('get_'+methods[i])
            if self.qle_ps[i].text() == '':
                best_p, a = get_best_params(data_train, data_test, method)
                r = classifier(data_train, data_test, method, best_p)
                self.qle_ps[i].setText(str(round(best_p, 2)))
                # self.qle_ps[i].setReadOnly(True)
                self.qle_acs[i].setText(str(round(a, 2)))
            else:
                if method == get_scale:
                    best_p = float(self.qle_ps[i].text())
                else:
                    best_p = int(self.qle_ps[i].text())
                r = classifier(data_train, data_test, method, best_p)
                a = accuracy_score(r, data_test[1])
                self.qle_acs[i].setText(str(round(a, 2)))
            result.append(r)

        result = np.array(result)
        result = result.transpose()
        result = result.tolist()
        result = list(map(lambda x: max(x, key=x.count), result))
        ac = accuracy_score(result, data_test[1])
        self.qle_acs[-1].setText(str(round(ac, 2)))

        fig = plt.figure(figsize=(12, 12))
        ax_im = []
        ax_f = []
        for i in range(6):
            ax_im.append(fig.add_subplot(3, 6, i+1))
            ax_f.append(fig.add_subplot(3, 6, i+7))
        ax = fig.add_subplot(3, 1, 3)
        ax.set_xlim(0, len(data_test[0]))
        ax.set_ylim(0, 1.2)
        plt.ion()
        res = {"histogram":[], "dft":[], "dct":[], "gradient":[], "scale":[], "voting":[]}

        for k in range(len(data_test[0])):
            if data_test[1][k] == result[k]:
                res['voting'].append(1)
            else:
                res['voting'].append(0)

        for k in range(len(data_test[0])):
            ax_im[0].clear()
            ax_im[0].set_title('Test image:')
            ax_f[0].clear()
            ax_f[0].set_title('Voting:')
            for i in range(1, 6):
                ax_im[i].clear()
                ax_im[i].set_title(methods[i-1] + ':')
                ax_f[i].clear()
            ax.clear()
            ax.set_xlabel('Test number')
            ax.set_ylabel('Accuracy')
            ax.set_xlim(0, len(data_test[0]))
            ax.set_ylim(0, 1.2)

            image = cv2.resize(data_test[0][k], (100, 100), interpolation=cv2.INTER_AREA)
            cv2.imwrite('test.jpg', 255 * image)
            image = plt.imread('test.jpg')
            ax_im[0].imshow(image, cmap='gray')

            # self.get_plot(get_dft, data_test[0][k], 10, ax_f[0])

            votings = []
            for i in range(5):
                method = eval('get_'+methods[i])
                if method == get_scale:
                    p = float(self.qle_ps[i].text())
                else:
                    p = int(self.qle_ps[i].text())

                ind = closest(data_train, data_test[0][k], method, p)
                votings.append(ind)
                image = cv2.resize(data_train[0][ind], (100, 100), interpolation=cv2.INTER_AREA)
                cv2.imwrite('test.jpg', 255 * image)
                image = plt.imread('test.jpg')
                ax_im[i+1].imshow(image, cmap='gray')
                self.get_plot(method, data_train[0][ind], p, ax_f[i+1])

                if data_test[1][k] == data_train[1][ind]:
                    res[methods[i]].append(1)
                else:
                    res[methods[i]].append(0)


            ind_vot = collections.Counter(set(votings)).most_common()
            print(ind_vot)
            image = cv2.resize(data_train[0][ind_vot[0][0]], (100, 100), interpolation=cv2.INTER_AREA)
            cv2.imwrite('test.jpg', 255 * image)
            image = plt.imread('test.jpg')
            ax_f[0].imshow(image, cmap='gray')


            if data_test[1][k] == result[k]:
                res['voting'].append(1)
            else:
                res['voting'].append(0)
            data = [mean(res['voting'][:i+1]) for i in range(len(res['voting']))]
            # for y in data:
            #     ax.plot([i for i in range(len(res['voting']))], y)
            ax.plot([i for i in range(len(data))], data)
            ax.legend(['voting'])#["histogram", "dft", "dct", "gradient", "scale", "voting"])
            plt.pause(2)
            fig.show()
            fig.canvas.draw()


    def on_click_start_5(self):
        plot = []
        data = get_data()
        ps = []
        for i in range(9, 0, -1):
            data_train, data_test = get_split_data(data, i)
            r, a, p = get_vote(data_train, data_test)
            ps.append(p)
            ac = accuracy_score(r, data_test[1])
            a.append(ac)
            plot.append(a)


        fin = []
        for i in range(len(plot)):
            fin.append(plot[i])
            fin.append(ps[i])


        for i in range(len(plot[0])-1):
            for j in range(0, 17, 2):
                self.qles[i][j].setText(str(round(fin[j][i], 2)))
                self.qles[i][j+1].setText(str(round(fin[j+1][i], 2)))

        for i in range(9):
            self.qle_a_f[i].setText(str(round(plot[i][-1], 2)))

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        # ax = fig.add_subplot(111)
        plot = np.array(plot)
        plot = plot.transpose()
        for i in plot:
            plt.plot([j for j in range(1, len(i) + 1)], i)
        plt.grid(True)
        plt.legend(["histogram", "dft", "dct", "gradient", "scale", "voting"])
        plt.title('cross validate')
        plt.xlabel("Number of train")
        plt.ylabel("Accuracy")
        plt.savefig('plot1.png')
        pixmap = QPixmap('plot1.png')
        self.lbl_final.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = mainWindow()
    sys.exit(app.exec_())