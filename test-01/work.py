import pandas as pd


class Base:
    def __call__(self, something):
        print(f"Base({something})")


class Abc(Base):
    def __init__(self):
        print(f"Abc()")


abc = Abc()

abc("Ulala")
