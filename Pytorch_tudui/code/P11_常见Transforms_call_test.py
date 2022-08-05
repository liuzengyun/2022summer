# from PIL import Image
# from torchvision import transforms
#
# img_path = "data/data/train/ants_image/0013035.jpg"
# img = Image.open(img_path)
# print(img)
class Person:
    def __call__(self, name):
        print('__call__'+' Hello '+name)

    def hello(self, name):
        print('hello '+name)


person = Person()
person.__call__("name1")
person('name2')
person.hello('name3')

