class Parent:
    def __init__(self):
        print("Parent's __init__")
        self.name = "Parent"

class Child(Parent):
    def __init__(self):
        super().__init__()  # Calls Parent's __init__ method
        print("Child's __init__")
        self.age = 10

class MyClass:
    def __init__(self, value):
        self.my_variable = value  # An instance variable

    def show(self):
        print(self.my_variable)  # Accessing an instance variable

obj = MyClass(5)
obj.show()  # Prints: 5

c = Child()