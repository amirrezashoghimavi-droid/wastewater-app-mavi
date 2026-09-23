import sys
import math
import re
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QLabel, QLineEdit, QPushButton, QTextEdit, QHBoxLayout,
                               QFrame, QGridLayout, QScrollArea, QSpinBox, QComboBox,
                               QTabWidget, QListWidget, QListWidgetItem, QStackedWidget)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont
import numpy as np
from sympy import Matrix, lcm


# ═══════════════════════════════════════════════════════════════
# توابع موازنه شیمیایی
# ═══════════════════════════════════════════════════════════════
def parse_compound(compound):
    atoms = {}
    pattern = r'([A-Z][a-z]*)(\d*)'
    for match in re.finditer(pattern, compound):
        element = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        atoms[element] = atoms.get(element, 0) + count
    return atoms


def balance_chemical_reaction(reactants, products):
    all_elements = set()
    for compound in reactants + products:
        all_elements.update(parse_compound(compound).keys())
    all_elements = list(all_elements)

    matrix = []
    for element in all_elements:
        row = []
        for compound in reactants:
            atoms = parse_compound(compound)
            row.append(atoms.get(element, 0))
        for compound in products:
            atoms = parse_compound(compound)
            row.append(-atoms.get(element, 0))
        matrix.append(row)

    A = Matrix(matrix)
    nullspace = A.nullspace()
    if not nullspace:
        return None

    solution = nullspace[0]
    lcm_denoms = 1
    for val in solution:
        if val != 0:
            lcm_denoms = lcm(lcm_denoms, val.q)
    coefficients = [int(val * lcm_denoms) for val in solution]
    return coefficients


# ═══════════════════════════════════════════════════════════════
# پنجره اصلی
# ═══════════════════════════════════════════════════════════════
class MegaChemicalSuite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚗️ ابربرنامه جامع سینتیک شیمیایی")
        self.setGeometry(50, 30, 1200, 750)

        self.setStyleSheet("""
            QMainWindow { background: #f8f9fa; }
            
            /* تب‌های عمودی سمت چپ */
            QListWidget#sidebar {
                background: #1a237e;
                border: none;
                border-radius: 12px;
                padding: 8px;
                outline: none;
                font-family: Tahoma;
            }
            QListWidget#sidebar::item {
                background: transparent;
                color: #b0bec5;
                padding: 14px 12px;
                margin: 3px 0;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }
            QListWidget#sidebar::item:hover {
                background: #283593;
                color: white;
            }
            QListWidget#sidebar::item:selected {
                background: #42a5f5;
                color: white;
                border-left: 4px solid #ffeb3b;
            }
            
            QLabel { color: #1a237e; font-size: 13px; font-weight: bold; padding: 4px; }
            QLineEdit {
                background: white; color: #1a237e; padding: 9px;
                border: 2px solid #9fa8da; border-radius: 8px; font-size: 13px;
            }
            QLineEdit:focus { border: 2px solid #3949ab; background: #f3f4ff; }
            QPushButton {
                background: #3949ab; color: white; padding: 11px 22px;
                border-radius: 9px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background: #283593; }
            QPushButton:pressed { background: #1a237e; }
            
            QFrame#card {
                background: #f3f4ff;
                border: 2px solid #c5cae9;
                border-radius: 12px;
                padding: 14px;
            }
            QFrame#resultPanel {
                background: #e8f5e9;
                border: 2px solid #a5d6a7;
                border-radius: 12px;
                padding: 14px;
            }
            QTextEdit {
                background: white; color: #1b5e20; 
                border: 2px solid #a5d6a7;
                border-radius: 10px; font-size: 13px; padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QScrollArea { border: none; background: transparent; }
            QSpinBox, QComboBox {
                background: white; color: #1a237e; padding: 7px;
                border: 2px solid #9fa8da; border-radius: 7px;
                font-size: 13px; font-weight: bold;
            }
            
            QLabel#pageTitle {
                font-size: 20px; color: #1a237e; font-weight: bold;
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e8eaf6, stop:1 #f8f9fa);
                border-radius: 8px;
                border-bottom: 3px solid #3949ab;
            }
            QLabel#panelTitle {
                font-size: 15px; color: #283593; font-weight: bold;
                padding: 8px;
                border-bottom: 2px solid #c5cae9;
            }
            QLabel#resultTitle {
                font-size: 15px; color: #2e7d32; font-weight: bold;
                padding: 8px;
                border-bottom: 2px solid #a5d6a7;
            }
        """)

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)
        central.setLayout(main_layout)

        # ═══════════════ سایدبار عمودی سمت چپ ═══════════════
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(230)
        self.sidebar.setIconSize(QSize(24, 24))

        # لیست تب‌ها
        tabs = [
            "⚖️  موازنه واکنش",
            "🔥  سرعت واکنش",
            "0️⃣  درجه صفر",
            "1️⃣  درجه اول",
            "2️⃣  درجه دوم",
            "🧪  اشباع (آنزیمی)",
            "🔥  آرنیوس",
            "🧪  استوکیومتری",
            "🧪  فرآیند هابر",
            "☢️  واپاشی رادیواکتیو",
            "☢️  نیمه‌عمر",
            "🧫  نیتریفیکاسیون",
            "🏭  لجن فعال (MLSS)",
            "🏭  تخلیه پساب",
            "🍬  تخمیر قند",
            "🧪  اکسیداسیون H₂S",
        ]
        for t in tabs:
            item = QListWidgetItem(t)
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.sidebar.addItem(item)

        # ═══════════════ محتوای سمت راست ═══════════════
        self.stack = QStackedWidget()

        # اضافه کردن صفحات
        self.stack.addWidget(self.create_balancer_page())
        self.stack.addWidget(self.create_rate_page())
        self.stack.addWidget(self.create_zero_page())
        self.stack.addWidget(self.create_first_page())
        self.stack.addWidget(self.create_second_page())
        self.stack.addWidget(self.create_saturation_page())
        self.stack.addWidget(self.create_arrhenius_page())
        self.stack.addWidget(self.create_stoichiometry_page())
        self.stack.addWidget(self.create_haber_page())
        self.stack.addWidget(self.create_radioactive_page())
        self.stack.addWidget(self.create_halflife_page())
        self.stack.addWidget(self.create_nitrification_page())
        self.stack.addWidget(self.create_mlss_page())
        self.stack.addWidget(self.create_brine_page())
        self.stack.addWidget(self.create_fermentation_page())
        self.stack.addWidget(self.create_h2s_page())

        # اتصال
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stack, 1)

    # ═══════════════════════════════════════════════════════════
    # صفحه‌ساز کمکی
    # ═══════════════════════════════════════════════════════════
    def make_page(self, title, input_widgets, calc_func):
        """ساخت یه صفحه کامل با ورودی سمت چپ و نتیجه سمت راست"""
        page = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        page.setLayout(layout)

        # عنوان
        lbl_title = QLabel(title)
        lbl_title.setObjectName("pageTitle")
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)

        # دو تا پنل کنار هم
        h_layout = QHBoxLayout()
        h_layout.setSpacing(12)

        # ── پنل ورودی (چپ) ──
        input_panel = QFrame()
        input_panel.setObjectName("card")
        input_layout = QVBoxLayout()
        input_panel.setLayout(input_layout)

        lbl_in = QLabel("📝 ورودی‌ها")
        lbl_in.setObjectName("panelTitle")
        input_layout.addWidget(lbl_in)

        for w in input_widgets:
            input_layout.addWidget(w)
        input_layout.addStretch()

        # ── پنل نتیجه (راست) ──
        result_panel = QFrame()
        result_panel.setObjectName("resultPanel")
        result_layout = QVBoxLayout()
        result_panel.setLayout(result_layout)

        lbl_out = QLabel("📊 نتیجه")
        lbl_out.setObjectName("resultTitle")
        result_layout.addWidget(lbl_out)

        result_text = QTextEdit()
        result_text.setReadOnly(True)
        result_text.setPlaceholderText("نتیجه اینجا نمایش داده می‌شود...")
        result_layout.addWidget(result_text)

        h_layout.addWidget(input_panel, 1)
        h_layout.addWidget(result_panel, 1)
        layout.addLayout(h_layout)

        # اتصال دکمه محاسبه
        for w in input_widgets:
            if isinstance(w, QPushButton):
                w.clicked.connect(lambda _, rt=result_text: calc_func(rt))

        return page

    # ═══════════════════════════════════════════════════════════
    # ۱. موازنه واکنش
    # ═══════════════════════════════════════════════════════════
    def create_balancer_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("⚖️ موازنه واکنش‌های شیمیایی")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        # ورودی
        input_panel = QFrame()
        input_panel.setObjectName("card")
        input_layout = QVBoxLayout()
        input_panel.setLayout(input_layout)

        input_layout.addWidget(QLabel("📝 ورودی‌ها"))
        input_layout.addWidget(QLabel("واکنش‌دهنده‌ها (با + جدا کن):"))
        self.bal_reactants = QLineEdit()
        self.bal_reactants.setPlaceholderText("مثلاً: H2 + O2")
        input_layout.addWidget(self.bal_reactants)

        input_layout.addWidget(QLabel("محصولات (با + جدا کن):"))
        self.bal_products = QLineEdit()
        self.bal_products.setPlaceholderText("مثلاً: H2O")
        input_layout.addWidget(self.bal_products)

        btn = QPushButton("⚖️ موازنه کن")
        btn.clicked.connect(self.calc_balancer)
        input_layout.addWidget(btn)
        input_layout.addStretch()

        # نتیجه
        result_panel = QFrame()
        result_panel.setObjectName("resultPanel")
        result_layout = QVBoxLayout()
        result_panel.setLayout(result_layout)
        result_layout.addWidget(QLabel("📊 نتیجه"))

        self.bal_result = QTextEdit()
        self.bal_result.setReadOnly(True)
        result_layout.addWidget(self.bal_result)

        h_layout.addWidget(input_panel, 1)
        h_layout.addWidget(result_panel, 1)
        layout.addLayout(h_layout)

        return page

    def calc_balancer(self):
        r_text = self.bal_reactants.text().strip()
        p_text = self.bal_products.text().strip()

        if not r_text or not p_text:
            self.bal_result.setText("❌ هر دو بخش را پر کن!")
            return

        reactants = [r.strip() for r in r_text.split('+')]
        products = [p.strip() for p in p_text.split('+')]

        try:
            coefficients = balance_chemical_reaction(reactants, products)
            if coefficients is None:
                self.bal_result.setText("❌ واکنش قابل موازنه نیست!")
                return

            parts = []
            for coef, comp in zip(coefficients[:len(reactants)], reactants):
                parts.append(f"{coef}{comp}" if coef != 1 else comp)
            left = " + ".join(parts)

            parts = []
            for coef, comp in zip(coefficients[len(reactants):], products):
                parts.append(f"{coef}{comp}" if coef != 1 else comp)
            right = " + ".join(parts)

            self.bal_result.setText(
                f"✅ واکنش موازنه شد:\n\n{left}  →  {right}")
        except Exception as e:
            self.bal_result.setText(f"❌ خطا: {e}")

    # ═══════════════════════════════════════════════════════════
    # ۲. سرعت واکنش
    # ═══════════════════════════════════════════════════════════
    def create_rate_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🔥 سرعت واکنش — r = k[A]^n")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        input_panel = QFrame()
        input_panel.setObjectName("card")
        input_layout = QVBoxLayout()
        input_panel.setLayout(input_layout)

        input_layout.addWidget(QLabel("🔬 ثابت سرعت k:"))
        self.k_input = QLineEdit()
        self.k_input.setPlaceholderText("مثلاً 0.05")
        input_layout.addWidget(self.k_input)

        input_layout.addWidget(QLabel("🧪 غلظت [A]:"))
        self.A_input = QLineEdit()
        self.A_input.setPlaceholderText("مثلاً 2.5")
        input_layout.addWidget(self.A_input)

        input_layout.addWidget(QLabel("📊 درجه واکنش n:"))
        self.n_combo = QComboBox()
        self.n_combo.addItems(["0", "1", "2", "3"])
        input_layout.addWidget(self.n_combo)

        btn = QPushButton("⚡ محاسبه سرعت")
        btn.clicked.connect(self.calc_rate)
        input_layout.addWidget(btn)
        input_layout.addStretch()

        result_panel = QFrame()
        result_panel.setObjectName("resultPanel")
        result_layout = QVBoxLayout()
        result_panel.setLayout(result_layout)
        result_layout.addWidget(QLabel("📊 نتیجه"))
        self.rate_result = QTextEdit()
        self.rate_result.setReadOnly(True)
        result_layout.addWidget(self.rate_result)

        h_layout.addWidget(input_panel, 1)
        h_layout.addWidget(result_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_rate(self):
        try:
            k = float(self.k_input.text())
            A = float(self.A_input.text())
            n = int(self.n_combo.currentText())
            r = k * (A ** n)
            self.rate_result.setText(
                f"📊 r = k[A]^n\n\n"
                f"r = {k} × ({A})^{n}\n"
                f"✅ r = {r:.6f} mol/L·t"
            )
        except ValueError:
            self.rate_result.setText("❌ لطفاً اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۳. درجه صفر
    # ═══════════════════════════════════════════════════════════
    def create_zero_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("0️⃣ واکنش درجه صفر — [A] = [A]₀ - kt")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        input_panel = QFrame()
        input_panel.setObjectName("card")
        input_layout = QVBoxLayout()
        input_panel.setLayout(input_layout)

        input_layout.addWidget(QLabel("🧪 غلظت اولیه [A]₀:"))
        self.z_A0 = QLineEdit()
        input_layout.addWidget(self.z_A0)

        input_layout.addWidget(QLabel("🔬 ثابت سرعت k:"))
        self.z_k = QLineEdit()
        input_layout.addWidget(self.z_k)

        input_layout.addWidget(QLabel("⏱️ زمان t:"))
        self.z_t = QLineEdit()
        input_layout.addWidget(self.z_t)

        btn = QPushButton("⚡ محاسبه درجه صفر")
        btn.clicked.connect(self.calc_zero)
        input_layout.addWidget(btn)
        input_layout.addStretch()

        result_panel = QFrame()
        result_panel.setObjectName("resultPanel")
        result_layout = QVBoxLayout()
        result_panel.setLayout(result_layout)
        result_layout.addWidget(QLabel("📊 نتیجه"))
        self.zero_result = QTextEdit()
        self.zero_result.setReadOnly(True)
        result_layout.addWidget(self.zero_result)

        h_layout.addWidget(input_panel, 1)
        h_layout.addWidget(result_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_zero(self):
        try:
            A0 = float(self.z_A0.text())
            k = float(self.z_k.text())
            t = float(self.z_t.text())
            A = A0 - k * t
            t_half = A0 / (2 * k)
            self.zero_result.setText(
                f"📊 واکنش درجه صفر\n\n"
                f"[A] = [A]₀ - kt\n"
                f"✅ [A] = {max(0, A):.4f} mol/L\n"
                f"⏱️ نیمه‌عمر: {t_half:.4f}"
            )
        except ValueError:
            self.zero_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۴. درجه اول
    # ═══════════════════════════════════════════════════════════
    def create_first_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("1️⃣ واکنش درجه اول — [A] = [A]₀ e^(-kt)")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        input_panel = QFrame()
        input_panel.setObjectName("card")
        input_layout = QVBoxLayout()
        input_panel.setLayout(input_layout)

        input_layout.addWidget(QLabel("🧪 غلظت اولیه [A]₀:"))
        self.f_A0 = QLineEdit()
        input_layout.addWidget(self.f_A0)

        input_layout.addWidget(QLabel("🔬 ثابت سرعت k:"))
        self.f_k = QLineEdit()
        input_layout.addWidget(self.f_k)

        input_layout.addWidget(QLabel("⏱️ زمان t:"))
        self.f_t = QLineEdit()
        input_layout.addWidget(self.f_t)

        btn = QPushButton("⚡ محاسبه درجه اول")
        btn.clicked.connect(self.calc_first)
        input_layout.addWidget(btn)
        input_layout.addStretch()

        result_panel = QFrame()
        result_panel.setObjectName("resultPanel")
        result_layout = QVBoxLayout()
        result_panel.setLayout(result_layout)
        result_layout.addWidget(QLabel("📊 نتیجه"))
        self.first_result = QTextEdit()
        self.first_result.setReadOnly(True)
        result_layout.addWidget(self.first_result)

        h_layout.addWidget(input_panel, 1)
        h_layout.addWidget(result_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_first(self):
        try:
            A0 = float(self.f_A0.text())
            k = float(self.f_k.text())
            t = float(self.f_t.text())
            A = A0 * math.exp(-k * t)
            t_half = math.log(2) / k
            self.first_result.setText(
                f"📊 واکنش درجه اول\n\n"
                f"[A] = [A]₀ e^(-kt)\n"
                f"✅ [A] = {A:.4f} mol/L\n"
                f"⏱️ نیمه‌عمر: {t_half:.4f}"
            )
        except ValueError:
            self.first_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۵. درجه دوم
    # ═══════════════════════════════════════════════════════════
    def create_second_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("2️⃣ واکنش درجه دوم — 1/[A] = 1/[A]₀ + kt")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        input_panel = QFrame()
        input_panel.setObjectName("card")
        input_layout = QVBoxLayout()
        input_panel.setLayout(input_layout)

        input_layout.addWidget(QLabel("🧪 غلظت اولیه [A]₀:"))
        self.s_A0 = QLineEdit()
        input_layout.addWidget(self.s_A0)

        input_layout.addWidget(QLabel("🔬 ثابت سرعت k:"))
        self.s_k = QLineEdit()
        input_layout.addWidget(self.s_k)

        input_layout.addWidget(QLabel("⏱️ زمان t:"))
        self.s_t = QLineEdit()
        input_layout.addWidget(self.s_t)

        btn = QPushButton("⚡ محاسبه درجه دوم")
        btn.clicked.connect(self.calc_second)
        input_layout.addWidget(btn)
        input_layout.addStretch()

        result_panel = QFrame()
        result_panel.setObjectName("resultPanel")
        result_layout = QVBoxLayout()
        result_panel.setLayout(result_layout)
        result_layout.addWidget(QLabel("📊 نتیجه"))
        self.second_result = QTextEdit()
        self.second_result.setReadOnly(True)
        result_layout.addWidget(self.second_result)

        h_layout.addWidget(input_panel, 1)
        h_layout.addWidget(result_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_second(self):
        try:
            A0 = float(self.s_A0.text())
            k = float(self.s_k.text())
            t = float(self.s_t.text())
            A = 1 / (1/A0 + k * t)
            t_half = 1 / (k * A0)
            self.second_result.setText(
                f"📊 واکنش درجه دوم\n\n"
                f"1/[A] = 1/[A]₀ + kt\n"
                f"✅ [A] = {A:.4f} mol/L\n"
                f"⏱️ نیمه‌عمر: {t_half:.4f}"
            )
        except ValueError:
            self.second_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۶. اشباع (آنزیمی)
    # ═══════════════════════════════════════════════════════════
    def create_saturation_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🧪 سینتیک اشباع (آنزیمی) — r = k[A]/(Ks+[A])")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # تب‌های فرعی افقی
        sub_tabs = QTabWidget()
        layout.addWidget(sub_tabs)

        # ── زیرتب مستقیم ──
        sub1 = QWidget()
        s1_layout = QHBoxLayout()
        sub1.setLayout(s1_layout)

        in1 = QFrame()
        in1.setObjectName("card")
        in1_lay = QVBoxLayout()
        in1.setLayout(in1_lay)
        in1_lay.addWidget(QLabel("k (حداکثر سرعت):"))
        self.sat_k = QLineEdit()
        in1_lay.addWidget(self.sat_k)
        in1_lay.addWidget(QLabel("Ks (نیمه اشباع):"))
        self.sat_Ks = QLineEdit()
        in1_lay.addWidget(self.sat_Ks)
        in1_lay.addWidget(QLabel("[A] غلظت:"))
        self.sat_A = QLineEdit()
        in1_lay.addWidget(self.sat_A)
        b1 = QPushButton("🧮 محاسبه r")
        b1.clicked.connect(self.calc_sat_direct)
        in1_lay.addWidget(b1)
        in1_lay.addStretch()

        out1 = QFrame()
        out1.setObjectName("resultPanel")
        out1_lay = QVBoxLayout()
        out1.setLayout(out1_lay)
        out1_lay.addWidget(QLabel("📊 نتیجه"))
        self.sat_direct_result = QTextEdit()
        self.sat_direct_result.setReadOnly(True)
        out1_lay.addWidget(self.sat_direct_result)

        s1_layout.addWidget(in1, 1)
        s1_layout.addWidget(out1, 1)

        # ── زیرتب تخمین ──
        sub2 = QWidget()
        s2_layout = QHBoxLayout()
        sub2.setLayout(s2_layout)

        in2 = QFrame()
        in2.setObjectName("card")
        in2_lay = QVBoxLayout()
        in2.setLayout(in2_lay)

        num_frame = QHBoxLayout()
        num_frame.addWidget(QLabel("🔢 تعداد داده:"))
        self.sat_num_spin = QSpinBox()
        self.sat_num_spin.setMinimum(2)
        self.sat_num_spin.setMaximum(50)
        self.sat_num_spin.setValue(4)
        self.sat_num_spin.valueChanged.connect(self.update_sat_inputs)
        num_frame.addWidget(self.sat_num_spin)
        num_frame.addStretch()
        in2_lay.addLayout(num_frame)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.sat_data_container = QWidget()
        self.sat_data_layout = QVBoxLayout()
        self.sat_data_container.setLayout(self.sat_data_layout)
        scroll.setWidget(self.sat_data_container)
        in2_lay.addWidget(scroll)

        self.sat_inputs = []
        self.update_sat_inputs()

        b2 = QPushButton("📈 تخمین k و Ks")
        b2.clicked.connect(self.calc_sat_linear)
        in2_lay.addWidget(b2)

        out2 = QFrame()
        out2.setObjectName("resultPanel")
        out2_lay = QVBoxLayout()
        out2.setLayout(out2_lay)
        out2_lay.addWidget(QLabel("📊 نتیجه"))
        self.sat_linear_result = QTextEdit()
        self.sat_linear_result.setReadOnly(True)
        out2_lay.addWidget(self.sat_linear_result)

        s2_layout.addWidget(in2, 1)
        s2_layout.addWidget(out2, 1)

        # ── زیرتب نیمه‌اشباع ──
        sub3 = QWidget()
        s3_layout = QHBoxLayout()
        sub3.setLayout(s3_layout)

        in3 = QFrame()
        in3.setObjectName("card")
        in3_lay = QVBoxLayout()
        in3.setLayout(in3_lay)
        in3_lay.addWidget(QLabel("k:"))
        self.sat_half_k = QLineEdit()
        in3_lay.addWidget(self.sat_half_k)
        in3_lay.addWidget(QLabel("Ks:"))
        self.sat_half_Ks = QLineEdit()
        in3_lay.addWidget(self.sat_half_Ks)
        b3 = QPushButton("🔬 محاسبه r در Ks")
        b3.clicked.connect(self.calc_sat_half)
        in3_lay.addWidget(b3)
        in3_lay.addStretch()

        out3 = QFrame()
        out3.setObjectName("resultPanel")
        out3_lay = QVBoxLayout()
        out3.setLayout(out3_lay)
        out3_lay.addWidget(QLabel("📊 نتیجه"))
        self.sat_half_result = QTextEdit()
        self.sat_half_result.setReadOnly(True)
        out3_lay.addWidget(self.sat_half_result)

        s3_layout.addWidget(in3, 1)
        s3_layout.addWidget(out3, 1)

        sub_tabs.addTab(sub1, "🧮 مستقیم")
        sub_tabs.addTab(sub2, "📈 تخمین")
        sub_tabs.addTab(sub3, "🔬 نیمه اشباع")

        return page

    def update_sat_inputs(self):
        for i in reversed(range(self.sat_data_layout.count())):
            w = self.sat_data_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.sat_inputs = []
        num = self.sat_num_spin.value()
        for i in range(num):
            row = QFrame()
            row.setObjectName("card")
            rl = QHBoxLayout()
            row.setLayout(rl)
            rl.addWidget(QLabel(f"[A]{i+1}:"))
            eA = QLineEdit()
            eA.setMaximumWidth(120)
            rl.addWidget(eA)
            rl.addWidget(QLabel(f"r{i+1}:"))
            er = QLineEdit()
            er.setMaximumWidth(120)
            rl.addWidget(er)
            rl.addStretch()
            self.sat_data_layout.addWidget(row)
            self.sat_inputs.append((eA, er))

    def calc_sat_direct(self):
        try:
            k = float(self.sat_k.text())
            Ks = float(self.sat_Ks.text())
            A = float(self.sat_A.text())
            r = k * A / (Ks + A)
            self.sat_direct_result.setText(
                f"📊 r = k[A] / (Ks + [A])\n\n"
                f"r = {k} × {A} / ({Ks} + {A})\n"
                f"✅ r = {r:.6f} mole/L·t\n\n"
                f"💡 حداکثر: {k} mole/L·t"
            )
        except ValueError:
            self.sat_direct_result.setText("❌ اعداد معتبر وارد کن!")

    def calc_sat_linear(self):
        try:
            data = []
            for eA, er in self.sat_inputs:
                if eA.text().strip() and er.text().strip():
                    data.append((float(eA.text()), float(er.text())))
            if len(data) < 2:
                self.sat_linear_result.setText("❌ حداقل ۲ جفت داده!")
                return
            x_vals = [1/A for A, r in data]
            y_vals = [1/r for A, r in data]
            n = len(x_vals)
            sum_x = sum(x_vals)
            sum_y = sum(y_vals)
            sum_xy = sum(x*y for x, y in zip(x_vals, y_vals))
            sum_x2 = sum(x*x for x in x_vals)
            slope = (n*sum_xy - sum_x*sum_y)/(n*sum_x2 - sum_x**2)
            intercept = (sum_y - slope*sum_x)/n
            k = 1/intercept
            Ks = slope * k
            self.sat_linear_result.setText(
                f"📈 Linearization: 1/r = 1/k + (Ks/k)(1/[A])\n\n"
                f"🔢 تعداد داده‌ها: {len(data)}\n\n"
                f"Slope = {slope:.6f}\n"
                f"Intercept = {intercept:.6f}\n\n"
                f"✅ k = {k:.6f} mole/L·t\n"
                f"✅ Ks = {Ks:.6f} mole/L"
            )
        except ValueError:
            self.sat_linear_result.setText("❌ اعداد معتبر وارد کن!")

    def calc_sat_half(self):
        try:
            k = float(self.sat_half_k.text())
            Ks = float(self.sat_half_Ks.text())
            r = k/2
            self.sat_half_result.setText(
                f"🔬 در [A] = Ks = {Ks}:\n\n"
                f"r = k/2 = {r:.4f} mole/L·t"
            )
        except ValueError:
            self.sat_half_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۷. آرنیوس
    # ═══════════════════════════════════════════════════════════
    def create_arrhenius_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🔥 معادله آرنیوس — k = A·e^(-E/RT)")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub_tabs = QTabWidget()
        layout.addWidget(sub_tabs)

        R = 8.314

        # ── محاسبه k ──
        s1 = QWidget()
        s1l = QHBoxLayout()
        s1.setLayout(s1l)

        in1 = QFrame()
        in1.setObjectName("card")
        i1l = QVBoxLayout()
        in1.setLayout(i1l)
        i1l.addWidget(QLabel("A (فاکتور فرکانس):"))
        self.arr_A = QLineEdit()
        self.arr_A.setPlaceholderText("1e10")
        i1l.addWidget(self.arr_A)
        i1l.addWidget(QLabel("E (انرژی فعال‌سازی) J/mol:"))
        self.arr_E = QLineEdit()
        self.arr_E.setPlaceholderText("50000")
        i1l.addWidget(self.arr_E)
        i1l.addWidget(QLabel("T (دما) K:"))
        self.arr_T = QLineEdit()
        self.arr_T.setPlaceholderText("298")
        i1l.addWidget(self.arr_T)
        b = QPushButton("🔥 محاسبه k")
        b.clicked.connect(lambda: self.calc_arr_k(R))
        i1l.addWidget(b)
        i1l.addStretch()

        out1 = QFrame()
        out1.setObjectName("resultPanel")
        o1l = QVBoxLayout()
        out1.setLayout(o1l)
        o1l.addWidget(QLabel("📊 نتیجه"))
        self.arr_k_result = QTextEdit()
        self.arr_k_result.setReadOnly(True)
        o1l.addWidget(self.arr_k_result)

        s1l.addWidget(in1, 1)
        s1l.addWidget(out1, 1)

        # ── محاسبه E ──
        s2 = QWidget()
        s2l = QHBoxLayout()
        s2.setLayout(s2l)

        in2 = QFrame()
        in2.setObjectName("card")
        i2l = QVBoxLayout()
        in2.setLayout(i2l)
        i2l.addWidget(QLabel("A:"))
        self.arrE_A = QLineEdit()
        i2l.addWidget(self.arrE_A)
        i2l.addWidget(QLabel("k:"))
        self.arrE_k = QLineEdit()
        i2l.addWidget(self.arrE_k)
        i2l.addWidget(QLabel("T (K):"))
        self.arrE_T = QLineEdit()
        i2l.addWidget(self.arrE_T)
        b = QPushButton("⚡ محاسبه E")
        b.clicked.connect(lambda: self.calc_arr_E(R))
        i2l.addWidget(b)
        i2l.addStretch()

        out2 = QFrame()
        out2.setObjectName("resultPanel")
        o2l = QVBoxLayout()
        out2.setLayout(o2l)
        o2l.addWidget(QLabel("📊 نتیجه"))
        self.arr_E_result = QTextEdit()
        self.arr_E_result.setReadOnly(True)
        o2l.addWidget(self.arr_E_result)

        s2l.addWidget(in2, 1)
        s2l.addWidget(out2, 1)

        # ── دو دما ──
        s3 = QWidget()
        s3l = QHBoxLayout()
        s3.setLayout(s3l)

        in3 = QFrame()
        in3.setObjectName("card")
        i3l = QVBoxLayout()
        in3.setLayout(i3l)
        for lbl, name in [("k1:", "tt_k1"), ("T1 (K):", "tt_T1"),
                          ("k2:", "tt_k2"), ("T2 (K):", "tt_T2")]:
            i3l.addWidget(QLabel(lbl))
            e = QLineEdit()
            setattr(self, name, e)
            i3l.addWidget(e)
        b = QPushButton("📊 محاسبه E")
        b.clicked.connect(lambda: self.calc_arr_2T(R))
        i3l.addWidget(b)
        i3l.addStretch()

        out3 = QFrame()
        out3.setObjectName("resultPanel")
        o3l = QVBoxLayout()
        out3.setLayout(o3l)
        o3l.addWidget(QLabel("📊 نتیجه"))
        self.arr_2T_result = QTextEdit()
        self.arr_2T_result.setReadOnly(True)
        o3l.addWidget(self.arr_2T_result)

        s3l.addWidget(in3, 1)
        s3l.addWidget(out3, 1)

        sub_tabs.addTab(s1, "🔥 محاسبه k")
        sub_tabs.addTab(s2, "⚡ محاسبه E")
        sub_tabs.addTab(s3, "📊 دو دما")

        return page

    def calc_arr_k(self, R):
        try:
            A = float(self.arr_A.text())
            E = float(self.arr_E.text())
            T = float(self.arr_T.text())
            k = A * math.exp(-E/(R*T))
            self.arr_k_result.setText(
                f"🔥 k = A·e^(-E/RT)\n\n"
                f"A = {A}\nE = {E}\nT = {T} K\nR = {R}\n\n"
                f"✅ k = {k:.6e}"
            )
        except ValueError:
            self.arr_k_result.setText("❌ اعداد معتبر وارد کن!")

    def calc_arr_E(self, R):
        try:
            A = float(self.arrE_A.text())
            k = float(self.arrE_k.text())
            T = float(self.arrE_T.text())
            E = -R*T*math.log(k/A)
            self.arr_E_result.setText(
                f"⚡ E = -RT·ln(k/A)\n\n"
                f"✅ E = {E:.2f} J/mol = {E/1000:.2f} kJ/mol"
            )
        except ValueError:
            self.arr_E_result.setText("❌ اعداد معتبر وارد کن!")

    def calc_arr_2T(self, R):
        try:
            k1 = float(self.tt_k1.text())
            T1 = float(self.tt_T1.text())
            k2 = float(self.tt_k2.text())
            T2 = float(self.tt_T2.text())
            E = R*math.log(k2/k1)/(1/T1 - 1/T2)
            self.arr_2T_result.setText(
                f"📊 E = R·ln(k2/k1)/(1/T1 - 1/T2)\n\n"
                f"✅ E = {E:.2f} J/mol = {E/1000:.2f} kJ/mol"
            )
        except ValueError:
            self.arr_2T_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۸. استوکیومتری
    # ═══════════════════════════════════════════════════════════
    def create_stoichiometry_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🧪 استوکیومتری و سینتیک — r = k[A]^α[B]^β")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        in_panel = QFrame()
        in_panel.setObjectName("card")
        in_lay = QVBoxLayout()
        in_panel.setLayout(in_lay)

        self.stoich_inputs = {}
        labels = [
            ("k (ثابت سرعت):", "k_input"),
            ("[A]:", "A_input"),
            ("α (توان A):", "alpha_input"),
            ("[B]:", "B_input"),
            ("β (توان B):", "beta_input"),
            ("ضریب a:", "a_input"),
            ("ضریب b:", "b_input"),
            ("ضریب c:", "c_input"),
            ("ضریب d:", "d_input"),
        ]
        for lbl, name in labels:
            in_lay.addWidget(QLabel(lbl))
            e = QLineEdit()
            in_lay.addWidget(e)
            self.stoich_inputs[name] = e

        b1 = QPushButton("🧮 محاسبه سرعت و درجه")
        b1.clicked.connect(self.calc_stoich)
        in_lay.addWidget(b1)
        b2 = QPushButton("📊 نسبت‌های واکنش")
        b2.clicked.connect(self.calc_stoich_ratios)
        in_lay.addWidget(b2)
        in_lay.addStretch()

        out_panel = QFrame()
        out_panel.setObjectName("resultPanel")
        out_lay = QVBoxLayout()
        out_panel.setLayout(out_lay)
        out_lay.addWidget(QLabel("📊 نتیجه"))
        self.stoich_result = QTextEdit()
        self.stoich_result.setReadOnly(True)
        out_lay.addWidget(self.stoich_result)

        h_layout.addWidget(in_panel, 1)
        h_layout.addWidget(out_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_stoich(self):
        try:
            v = {n: float(e.text()) for n, e in self.stoich_inputs.items()}
            r = v['k_input'] * (v['A_input'] ** v['alpha_input']
                                ) * (v['B_input'] ** v['beta_input'])
            total = v['alpha_input'] + v['beta_input']
            if total == 0:
                ku = "mole/L·t"
            elif total == 1:
                ku = "t⁻¹"
            elif total == 2:
                ku = "L/mole·t"
            elif total == 3:
                ku = "L²/mole²·t"
            else:
                ku = f"(mole/L)^(1-{int(total)})·t⁻¹"
            self.stoich_result.setText(
                f"r = k[A]^α[B]^β\n"
                f"r = {v['k_input']} × ({v['A_input']})^{int(v['alpha_input'])} × ({v['B_input']})^{int(v['beta_input'])}\n"
                f"✅ r = {r:.6f} mole/L·t\n\n"
                f"📊 درجه کل: {int(total)}\n"
                f"🔬 واحد k: {ku}"
            )
        except ValueError:
            self.stoich_result.setText("❌ اعداد معتبر وارد کن!")

    def calc_stoich_ratios(self):
        try:
            v = {n: float(e.text()) for n, e in self.stoich_inputs.items()}
            self.stoich_result.setText(
                f"📊 روابط سرعت‌ها:\n\n"
                f"r = |rA|/{int(v['a_input'])} = |rB|/{int(v['b_input'])} = rC/{int(v['c_input'])} = rD/{int(v['d_input'])}\n\n"
                f"• |rA|/rC = a/c = {int(v['a_input'])}/{int(v['c_input'])}\n"
                f"• |rB|/rC = b/c = {int(v['b_input'])}/{int(v['c_input'])}"
            )
        except ValueError:
            self.stoich_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۹. فرآیند هابر
    # ═══════════════════════════════════════════════════════════
    def create_haber_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🧪 فرآیند هابر — N₂ + 3H₂ → 2NH₃")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        in_panel = QFrame()
        in_panel.setObjectName("card")
        in_lay = QVBoxLayout()
        in_panel.setLayout(in_lay)

        in_lay.addWidget(QLabel("نرخ تولید NH₃ (mole/L·s):"))
        self.haber_rate = QLineEdit()
        self.haber_rate.setPlaceholderText("2.0e-4")
        in_lay.addWidget(self.haber_rate)

        in_lay.addWidget(QLabel("ضریب N₂:"))
        self.haber_a = QLineEdit("1")
        in_lay.addWidget(self.haber_a)

        in_lay.addWidget(QLabel("ضریب H₂:"))
        self.haber_b = QLineEdit("3")
        in_lay.addWidget(self.haber_b)

        in_lay.addWidget(QLabel("ضریب NH₃:"))
        self.haber_c = QLineEdit("2")
        in_lay.addWidget(self.haber_c)

        b = QPushButton("🧮 محاسبه")
        b.clicked.connect(self.calc_haber)
        in_lay.addWidget(b)
        in_lay.addStretch()

        out_panel = QFrame()
        out_panel.setObjectName("resultPanel")
        out_lay = QVBoxLayout()
        out_panel.setLayout(out_lay)
        out_lay.addWidget(QLabel("📊 نتیجه"))
        self.haber_result = QTextEdit()
        self.haber_result.setReadOnly(True)
        out_lay.addWidget(self.haber_result)

        h_layout.addWidget(in_panel, 1)
        h_layout.addWidget(out_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_haber(self):
        try:
            rNH3 = float(self.haber_rate.text())
            a = float(self.haber_a.text())
            b = float(self.haber_b.text())
            c = float(self.haber_c.text())
            rN2 = (a/c)*rNH3
            rH2 = (b/c)*rNH3
            self.haber_result.setText(
                f"🧪 فرآیند هابر: {int(a)}N₂ + {int(b)}H₂ → {int(c)}NH₃\n\n"
                f"📊 نرخ NH₃ = {rNH3:.2e}\n\n"
                f"1️⃣ -d[N₂]/dt = {rN2:.2e} mole/L·s\n"
                f"2️⃣ -d[H₂]/dt = {rH2:.2e} mole/L·s"
            )
        except ValueError:
            self.haber_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۱۰. واپاشی رادیواکتیو
    # ═══════════════════════════════════════════════════════════
    def create_radioactive_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("☢️ واپاشی رادیواکتیو — C = C₀·e^(-kt)")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        in_panel = QFrame()
        in_panel.setObjectName("card")
        in_lay = QVBoxLayout()
        in_panel.setLayout(in_lay)

        in_lay.addWidget(QLabel("نیمه‌عمر (سال):"))
        self.rad_t_half = QLineEdit()
        self.rad_t_half.setPlaceholderText("28")
        in_lay.addWidget(self.rad_t_half)

        in_lay.addWidget(QLabel("درصد کاهش (%):"))
        self.rad_reduction = QLineEdit()
        self.rad_reduction.setPlaceholderText("99")
        in_lay.addWidget(self.rad_reduction)

        b = QPushButton("🧮 محاسبه زمان")
        b.clicked.connect(self.calc_rad)
        in_lay.addWidget(b)
        in_lay.addStretch()

        out_panel = QFrame()
        out_panel.setObjectName("resultPanel")
        out_lay = QVBoxLayout()
        out_panel.setLayout(out_lay)
        out_lay.addWidget(QLabel("📊 نتیجه"))
        self.rad_result = QTextEdit()
        self.rad_result.setReadOnly(True)
        out_lay.addWidget(self.rad_result)

        h_layout.addWidget(in_panel, 1)
        h_layout.addWidget(out_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_rad(self):
        try:
            t_half = float(self.rad_t_half.text())
            reduction = float(self.rad_reduction.text())/100
            k = 0.693/t_half
            remaining = 1-reduction
            t = -math.log(remaining)/k
            self.rad_result.setText(
                f"☢️ محاسبه واپاشی:\n\n"
                f"k = 0.693/{t_half} = {k:.4f} yr⁻¹\n"
                f"C/C₀ = {remaining:.4f}\n"
                f"✅ t = {t:.1f} سال"
            )
        except ValueError:
            self.rad_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۱۱. نیمه‌عمر
    # ═══════════════════════════════════════════════════════════
    def create_halflife_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("☢️ نیمه‌عمر و تجزیه مواد خطرناک")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        in_panel = QFrame()
        in_panel.setObjectName("card")
        in_lay = QVBoxLayout()
        in_panel.setLayout(in_lay)

        for lbl, name, ph in [
            ("نیمه‌عمر (ساعت):", "hl_thalf", "12"),
            ("درصد باقیمانده ۱:", "hl_r1", "40"),
            ("درصد باقیمانده ۲:", "hl_r2", "80"),
            ("درصد تجزیه ۱:", "hl_d1", "40"),
            ("درصد تجزیه ۲:", "hl_d2", "80"),
        ]:
            in_lay.addWidget(QLabel(lbl))
            e = QLineEdit()
            e.setPlaceholderText(ph)
            in_lay.addWidget(e)
            setattr(self, name, e)

        b = QPushButton("🧮 محاسبه زمان‌ها")
        b.clicked.connect(self.calc_hl)
        in_lay.addWidget(b)
        in_lay.addStretch()

        out_panel = QFrame()
        out_panel.setObjectName("resultPanel")
        out_lay = QVBoxLayout()
        out_panel.setLayout(out_lay)
        out_lay.addWidget(QLabel("📊 نتیجه"))
        self.hl_result = QTextEdit()
        self.hl_result.setReadOnly(True)
        out_lay.addWidget(self.hl_result)

        h_layout.addWidget(in_panel, 1)
        h_layout.addWidget(out_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_hl(self):
        try:
            th = float(self.hl_thalf.text())
            r1 = float(self.hl_r1.text())/100
            r2 = float(self.hl_r2.text())/100
            d1 = float(self.hl_d1.text())/100
            d2 = float(self.hl_d2.text())/100
            k = 0.693/th
            t_r1 = -math.log(r1)/k
            t_r2 = -math.log(r2)/k
            t_d1 = -math.log(1-d1)/k
            t_d2 = -math.log(1-d2)/k
            self.hl_result.setText(
                f"☢️ نیمه‌عمر:\n\n"
                f"k = {k:.4f} h⁻¹\n\n"
                f"1️⃣ زمان {r1*100:.0f}% باقیمانده: {t_r1:.2f} h\n"
                f"2️⃣ زمان {r2*100:.0f}% باقیمانده: {t_r2:.2f} h\n"
                f"3️⃣ زمان {d1*100:.0f}% تجزیه: {t_d1:.2f} h\n"
                f"4️⃣ زمان {d2*100:.0f}% تجزیه: {t_d2:.2f} h"
            )
        except ValueError:
            self.hl_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۱۲. نیتریفیکاسیون
    # ═══════════════════════════════════════════════════════════
    def create_nitrification_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🧫 نیتریفیکاسیون — NH₃-N → NO₂-N → NO₃-N")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        in_panel = QFrame()
        in_panel.setObjectName("card")
        in_lay = QVBoxLayout()
        in_panel.setLayout(in_lay)

        for lbl, name, ph in [
            ("k1 (h⁻¹):", "nit_k1", "0.1"),
            ("k2 (h⁻¹):", "nit_k2", "0.05"),
            ("C₀(NH₃-N) mg/L:", "nit_C0NH3", "10"),
            ("C₀(NO₂-N) mg/L:", "nit_C0NO2", "0"),
            ("C₀(NO₃-N) mg/L:", "nit_C0NO3", "0"),
            ("زمان (h):", "nit_t", "24"),
        ]:
            in_lay.addWidget(QLabel(lbl))
            e = QLineEdit()
            e.setPlaceholderText(ph)
            in_lay.addWidget(e)
            setattr(self, name, e)

        b = QPushButton("🧮 محاسبه")
        b.clicked.connect(self.calc_nit)
        in_lay.addWidget(b)
        in_lay.addStretch()

        out_panel = QFrame()
        out_panel.setObjectName("resultPanel")
        out_lay = QVBoxLayout()
        out_panel.setLayout(out_lay)
        out_lay.addWidget(QLabel("📊 نتیجه"))
        self.nit_result = QTextEdit()
        self.nit_result.setReadOnly(True)
        out_lay.addWidget(self.nit_result)

        h_layout.addWidget(in_panel, 1)
        h_layout.addWidget(out_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_nit(self):
        try:
            k1 = float(self.nit_k1.text())
            k2 = float(self.nit_k2.text())
            C0NH3 = float(self.nit_C0NH3.text())
            C0NO2 = float(self.nit_C0NO2.text())
            C0NO3 = float(self.nit_C0NO3.text())
            t = float(self.nit_t.text())

            C_NH3 = C0NH3 * math.exp(-k1*t)
            if abs(k2-k1) < 1e-10:
                C_NO2 = k1*C0NH3*t*math.exp(-k1*t) + C0NO2*math.exp(-k2*t)
                C_NO3 = C0NH3*(1-math.exp(-k1*t)-k1*t*math.exp(-k1*t)
                               ) + C0NO2*(1-math.exp(-k2*t)) + C0NO3
            else:
                C_NO2 = (k1*C0NH3/(k2-k1))*(math.exp(-k1*t) -
                                            math.exp(-k2*t)) + C0NO2*math.exp(-k2*t)
                C_NO3 = C0NH3*(1-(k2*math.exp(-k1*t)-k1*math.exp(-k2*t)) /
                               (k2-k1)) + C0NO2*(1-math.exp(-k2*t)) + C0NO3

            self.nit_result.setText(
                f"🧫 حل در t = {t} h:\n\n"
                f"1️⃣ [NH₃-N] = {C_NH3:.4f} mg/L\n"
                f"2️⃣ [NO₂-N] = {C_NO2:.4f} mg/L\n"
                f"3️⃣ [NO₃-N] = {C_NO3:.4f} mg/L\n\n"
                f"✅ جمع: {C_NH3+C_NO2+C_NO3:.4f} mg/L"
            )
        except ValueError:
            self.nit_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۱۳. لجن فعال (MLSS)
    # ═══════════════════════════════════════════════════════════
    def create_mlss_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🏭 لجن فعال — MLSS و جریان برگشتی")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        in_panel = QFrame()
        in_panel.setObjectName("card")
        in_lay = QVBoxLayout()
        in_panel.setLayout(in_lay)

        for lbl, name, ph in [
            ("دبی ورودی Q_inf (m³/d):", "mlss_Qinf", "4500"),
            ("MLSS (mg/L):", "mlss_MLSS", "2500"),
            ("TSS برگشتی (mg/L):", "mlss_TSSras", "10000"),
            ("TSS ورودی (mg/L):", "mlss_TSSinf", "0"),
        ]:
            in_lay.addWidget(QLabel(lbl))
            e = QLineEdit()
            e.setPlaceholderText(ph)
            in_lay.addWidget(e)
            setattr(self, name, e)

        b = QPushButton("🧮 محاسبه")
        b.clicked.connect(self.calc_mlss)
        in_lay.addWidget(b)
        in_lay.addStretch()

        out_panel = QFrame()
        out_panel.setObjectName("resultPanel")
        out_lay = QVBoxLayout()
        out_panel.setLayout(out_lay)
        out_lay.addWidget(QLabel("📊 نتیجه"))
        self.mlss_result = QTextEdit()
        self.mlss_result.setReadOnly(True)
        out_lay.addWidget(self.mlss_result)

        h_layout.addWidget(in_panel, 1)
        h_layout.addWidget(out_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_mlss(self):
        try:
            Qi = float(self.mlss_Qinf.text())
            MLSS = float(self.mlss_MLSS.text())
            TSSr = float(self.mlss_TSSras.text())
            TSSi = float(self.mlss_TSSinf.text())
            Qras = (MLSS*Qi - TSSi*Qi)/(TSSr - MLSS)
            Rras = Qras/Qi
            self.mlss_result.setText(
                f"🏭 موازنه جرم:\n\n"
                f"✅ Q_ras = {Qras:.1f} m³/d\n"
                f"✅ R_ras = {Rras:.3f}\n"
                f"✅ Q_total = {Qi+Qras:.1f} m³/d"
            )
        except (ValueError, ZeroDivisionError):
            self.mlss_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۱۴. تخلیه پساب
    # ═══════════════════════════════════════════════════════════
    def create_brine_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🏭 تخلیه پساب صنعتی به رودخانه")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        in_panel = QFrame()
        in_panel.setObjectName("card")
        in_lay = QVBoxLayout()
        in_panel.setLayout(in_lay)

        for lbl, name, ph in [
            ("TDS پساب (mg/L):", "br_Cbrine", "15600"),
            ("حد مجاز TDS (mg/L):", "br_Callow", "500"),
            ("دبی رودخانه (m³/d):", "br_Qstream", "8500"),
            ("TDS زمینه (mg/L):", "br_Cstream", "210"),
        ]:
            in_lay.addWidget(QLabel(lbl))
            e = QLineEdit()
            e.setPlaceholderText(ph)
            in_lay.addWidget(e)
            setattr(self, name, e)

        b = QPushButton("🧮 محاسبه دبی مجاز")
        b.clicked.connect(self.calc_brine)
        in_lay.addWidget(b)
        in_lay.addStretch()

        out_panel = QFrame()
        out_panel.setObjectName("resultPanel")
        out_lay = QVBoxLayout()
        out_panel.setLayout(out_lay)
        out_lay.addWidget(QLabel("📊 نتیجه"))
        self.brine_result = QTextEdit()
        self.brine_result.setReadOnly(True)
        out_lay.addWidget(self.brine_result)

        h_layout.addWidget(in_panel, 1)
        h_layout.addWidget(out_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_brine(self):
        try:
            Cb = float(self.br_Cbrine.text())
            Ca = float(self.br_Callow.text())
            Qs = float(self.br_Qstream.text())
            Cs = float(self.br_Cstream.text())
            Qb = Qs*(Ca - Cs)/(Cb - Ca)
            self.brine_result.setText(
                f"🏭 موازنه جرم:\n\n"
                f"✅ Q_brine = {Qb:.1f} m³/d\n"
                f"✅ Q_mix = {Qs+Qb:.1f} m³/d"
            )
        except (ValueError, ZeroDivisionError):
            self.brine_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۱۵. تخمیر قند
    # ═══════════════════════════════════════════════════════════
    def create_fermentation_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🍬 تخمیر قند — واکنش درجه صفر")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        in_panel = QFrame()
        in_panel.setObjectName("card")
        in_lay = QVBoxLayout()
        in_panel.setLayout(in_lay)

        for lbl, name, ph in [
            ("C₀ (mole/L):", "fer_C0", "0.15"),
            ("درصد تبدیل (%):", "fer_pct", "10"),
            ("زمان اول (h):", "fer_t1", "4"),
            ("زمان دوم (h):", "fer_t2", "24"),
        ]:
            in_lay.addWidget(QLabel(lbl))
            e = QLineEdit()
            e.setPlaceholderText(ph)
            in_lay.addWidget(e)
            setattr(self, name, e)

        b = QPushButton("🧮 محاسبه")
        b.clicked.connect(self.calc_fer)
        in_lay.addWidget(b)
        in_lay.addStretch()

        out_panel = QFrame()
        out_panel.setObjectName("resultPanel")
        out_lay = QVBoxLayout()
        out_panel.setLayout(out_lay)
        out_lay.addWidget(QLabel("📊 نتیجه"))
        self.fer_result = QTextEdit()
        self.fer_result.setReadOnly(True)
        out_lay.addWidget(self.fer_result)

        h_layout.addWidget(in_panel, 1)
        h_layout.addWidget(out_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_fer(self):
        try:
            C0 = float(self.fer_C0.text())
            pct = float(self.fer_pct.text())
            t1 = float(self.fer_t1.text())
            t2 = float(self.fer_t2.text())
            C1 = (1-pct/100)*C0
            k = (C0-C1)/t1
            C2 = max(0, C0 - k*t2)
            th = C0/(2*k)
            self.fer_result.setText(
                f"🍬 تخمیر قند:\n\n"
                f"C بعد از {t1}h = {C1:.4f} mole/L\n"
                f"✅ k = {k:.6f} mole/L·h\n"
                f"C بعد از {t2}h = {C2:.4f} mole/L\n"
                f"✅ t½ = {th:.2f} ساعت"
            )
        except (ValueError, ZeroDivisionError):
            self.fer_result.setText("❌ اعداد معتبر وارد کن!")

    # ═══════════════════════════════════════════════════════════
    # ۱۶. اکسیداسیون H₂S
    # ═══════════════════════════════════════════════════════════
    def create_h2s_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        title = QLabel("🧪 اکسیداسیون H₂S — شبه درجه اول")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        h_layout = QHBoxLayout()

        in_panel = QFrame()
        in_panel.setObjectName("card")
        in_lay = QVBoxLayout()
        in_panel.setLayout(in_lay)

        for lbl, name, ph in [
            ("k (L/mole·d):", "h2s_k", "1000"),
            ("[O₂] (mg/L):", "h2s_O2", "2"),
            ("MW O₂:", "h2s_MWO2", "32"),
            ("H₂S اولیه (mg/L):", "h2s_ini", "17"),
            ("H₂S نهایی (mg/L):", "h2s_fin", "0.034"),
            ("MW H₂S:", "h2s_MWH2S", "34"),
        ]:
            in_lay.addWidget(QLabel(lbl))
            e = QLineEdit()
            e.setPlaceholderText(ph)
            in_lay.addWidget(e)
            setattr(self, name, e)

        b = QPushButton("🧮 محاسبه زمان")
        b.clicked.connect(self.calc_h2s)
        in_lay.addWidget(b)
        in_lay.addStretch()

        out_panel = QFrame()
        out_panel.setObjectName("resultPanel")
        out_lay = QVBoxLayout()
        out_panel.setLayout(out_lay)
        out_lay.addWidget(QLabel("📊 نتیجه"))
        self.h2s_result = QTextEdit()
        self.h2s_result.setReadOnly(True)
        out_lay.addWidget(self.h2s_result)

        h_layout.addWidget(in_panel, 1)
        h_layout.addWidget(out_panel, 1)
        layout.addLayout(h_layout)
        return page

    def calc_h2s(self):
        try:
            k = float(self.h2s_k.text())
            O2 = float(self.h2s_O2.text())
            MWO2 = float(self.h2s_MWO2.text())
            Hi = float(self.h2s_ini.text())
            Hf = float(self.h2s_fin.text())
            MWH2S = float(self.h2s_MWH2S.text())

            O2_m = O2/1000/MWO2
            kp = k * O2_m
            Hi_m = Hi/1000/MWH2S
            Hf_m = Hf/1000/MWH2S
            t = -math.log(Hf_m/Hi_m)/kp

            self.h2s_result.setText(
                f"🧪 اکسیداسیون H₂S:\n\n"
                f"[O₂] = {O2_m:.6f} mole/L\n"
                f"✅ k' = {kp:.4f} d⁻¹\n"
                f"[H₂S]₀ = {Hi_m:.2e} mole/L\n"
                f"[H₂S] = {Hf_m:.2e} mole/L\n\n"
                f"✅ t = {t:.1f} روز"
            )
        except (ValueError, ZeroDivisionError):
            self.h2s_result.setText("❌ اعداد معتبر وارد کن!")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MegaChemicalSuite()
    window.show()
    sys.exit(app.exec())
