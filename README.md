# Knowledge-Distillation-Wheel #

This is a TensorFlow implementation of [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).

## Requirements

This model is almost self-contained, but to run it you still need: 

* Python 3.5+
* TensorFlow 1.7+
* NumPy
* SciPy
* Six

## Usage

### About The Dataset
Currently we only support MNIST, CIFAR-10 and CIFAR-100. For other datasets you should implement data decoder for yourself. More details please see `DataManager` class and `/data` dir.

### To Train Your Teacher Model
You can train your teacher model by few lines of command, more details please see `KDModel` class
```python
from model.KD_model import KDModel
from YOURMODELFILE import YourTeacherModel, YourStudentModel

YourModelName = "xxx"
YourDatasetName = "xxx"

kd_model = KDModel(YourDatasetName, YourTeacherModel, YourStudentModel)
kd_model.TrainTeacher(YourModelName, num_epoch=20)
kd_model.TestTeacher(YourModelName)
```

### To Train Your Student Model
You can easily train your student model using the output of your teacher model.
```python
from model.KD_model import KDModel
from model.teacher_model import mnist_teacher_model
from model.students_model import mnist_student_model

m = KDModel("MNIST", mnist_teacher_model, mnist_student_model)
m.TrainStudent(model_name="MNIST_test_student", teacher_model_name="test_MNIST", num_epoch=20, verbose=True)
m.TestStudent(model_name="MNIST_test_student", verbose=True)

```
