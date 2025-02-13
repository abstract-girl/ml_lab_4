from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    """
    Класс для вычисления длины шага.

    Parameters
    ----------
    lambda_ : float, optional
        Начальная скорость обучения. По умолчанию 1e-3.
    s0 : float, optional
        Параметр для вычисления скорости обучения. По умолчанию 1.
    p : float, optional
        Степенной параметр для вычисления скорости обучения. По умолчанию 0.5.
    iteration : int, optional
        Текущая итерация. По умолчанию 0.

    Methods
    -------
    __call__()
        Вычисляет скорость обучения на текущей итерации.
    """
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Вычисляет скорость обучения по формуле lambda * (s0 / (s0 + t))^p.

        Returns
        -------
        float
            Скорость обучения на текущем шаге.
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    """
    Перечисление для выбора функции потерь.

    Attributes
    ----------
    MSE : auto
        Среднеквадратическая ошибка.
    MAE : auto
        Средняя абсолютная ошибка.
    LogCosh : auto
        Логарифм гиперболического косинуса от ошибки.
    Huber : auto
        Функция потерь Хьюбера.
    """
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    Базовый класс для всех методов градиентного спуска.

    Parameters
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float, optional
        Параметр скорости обучения. По умолчанию 1e-3.
    loss_function : LossFunction, optional
        Функция потерь, которая будет оптимизироваться. По умолчанию MSE.

    Attributes
    ----------
    w : np.ndarray
        Вектор весов модели.
    lr : LearningRate
        Скорость обучения.
    loss_function : LossFunction
        Функция потерь.

    Methods
    -------
    step(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Шаг градиентного спуска.
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов на основе градиента. Метод шаблон.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам. Метод шаблон.
    calc_loss(x: np.ndarray, y: np.ndarray) -> float
        Вычисление значения функции потерь.
    predict(x: np.ndarray) -> np.ndarray
        Вычисление прогнозов на основе признаков x.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация базового класса для градиентного спуска.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Функция потерь, которая будет оптимизирована.

        Attributes
        ----------
        w : np.ndarray
            Начальный вектор весов, инициализированный случайным образом.
        lr : LearningRate
            Экземпляр класса для вычисления скорости обучения.
        loss_function : LossFunction
            Выбранная функция потерь.
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Выполнение одного шага градиентного спуска.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами.
        """

        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для обновления весов. Должен быть переопределен в подклассах.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для вычисления градиента функции потерь по весам. Должен быть переопределен в подклассах.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление значения функции потерь с использованием текущих весов.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        float
            Значение функции потерь.
        """
        y_pred = self.predict(x)  # Вычисление прогнозируемых значений

        if self.loss_function == LossFunction.MSE:
            errors = y - y_pred  # Разница между истинными значениями и прогнозами
            return np.mean(errors ** 2)  # Среднеквадратичная ошибка (MSE)
        elif self.loss_function == LossFunction.LogCosh:
            log_cosh = np.log(np.cosh(y_pred - y))
            return np.mean(log_cosh)
        else:
            return -1

        #raise NotImplementedError('BaseDescent calc_loss function not implemented')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Расчет прогнозов на основе признаков x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        np.ndarray
            Прогнозируемые значения.
        """
         # Добавляем колонку из единиц для учета свободного члена (w0) в векторе весов
        y_pred = x.dot(self.w) 
        return y_pred

        #raise NotImplementedError('BaseDescent predict function not implemented')


class VanillaGradientDescent(BaseDescent):

    def __init__(self,dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):

        super().__init__(dimension, lambda_, loss_function)
        self.k = 0  # Счетчик итераций
    """
    Класс полного градиентного спуска.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с учетом градиента.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам.
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов на основе градиента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        eta_k = self.lr.lambda_ * (self.lr.s0 / (self.lr.s0 + self.k)) ** self.lr.p # Вычисляем длину шага
        weight_difference = -eta_k * gradient  # Вычисляем изменение весов
        self.w += weight_difference  # Обновляем веса
        self.k += 1 
        return weight_difference

        #raise NotImplementedError('VanillaGradientDescent update_weights function not implemented')

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам.
        """
        # if x.shape[1] != self.w.shape[0]:  # Проверяем, соответствует ли количество признаков размеру вектора весов
        #     x = np.hstack([x, np.ones((x.shape[0], 1))])  # Добавляем столбец из единиц
        # # Проверяем, содержит ли x столбец для свободного члена. Если нет, добавляем его.
        # gradient = 2 / x.shape[0] * x.T.dot(x.dot(self.w) - y)
        # return gradient

        if self.loss_function == LossFunction.MSE:
            predictions = self.predict(x)
            error = predictions - y
            gradient = 2 * x.T.dot(error) / len(y)
        elif self.loss_function == LossFunction.LogCosh:
            predictions = self.predict(x)
            error = predictions - y
            gradient = -x.T.dot(np.tanh(error)) / len(y)
        else:
            raise NotImplementedError("Данный тип функции потерь не поддерживается")
        return gradient

    
        #raise NotImplementedError('VanillaGradientDescent calc_gradient function not implemented')


class StochasticDescent(VanillaGradientDescent):
    """
    Класс стохастического градиентного спуска.

    Parameters
    ----------
    batch_size : int, optional
        Размер мини-пакета. По умолчанию 50.

    Attributes
    ----------
    batch_size : int
        Размер мини-пакета.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по мини-пакетам.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):

        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по мини-пакетам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам, вычисленный по мини-пакету.
        """
        if self.loss_function == LossFunction.MSE:
            batch_indices = np.random.randint(0, x.shape[0], size=self.batch_size)
            x_batch = x[batch_indices]
            y_batch = y.iloc[batch_indices]
             # Вычисляем предсказания модели для мини-пакета
            y_pred_batch = self.predict(x_batch)
            # Вычисляем ошибку предсказания для мини-пакета
            error_batch = y_pred_batch - y_batch
            # Вычисляем градиент для мини-пакета
            gradient = 2 * x_batch.T.dot(error_batch) / self.batch_size
        elif self.loss_function == LossFunction.LogCosh:
            batch_indices = np.random.randint(0, x.shape[0], size=self.batch_size)
            x_batch = x[batch_indices]
            y_batch = y.iloc[batch_indices]
            y_pred_batch = self.predict(x_batch)
            error_batch = y_pred_batch - y_batch
            gradient = -x_batch.T.dot(np.tanh(error_batch)) / self.batch_size 
        else:
            raise NotImplementedError("Данный тип функции потерь не поддерживается")
        return gradient
        #raise NotImplementedError('StochasticDescent calc_gradient function not implemented')


class MomentumDescent(VanillaGradientDescent):
    """
    Класс градиентного спуска с моментом.

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    alpha : float
        Коэффициент момента.
    h : np.ndarray
        Вектор момента для весов.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием момента.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация класса градиентного спуска с моментом.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием момента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        eta_k = self.lr.lambda_  # Вычисляем длину шага
        # Обновляем вектор момента
        self.h = self.alpha * self.h + eta_k * gradient
        # Обновляем веса
        weight_difference = -self.h
        self.w += weight_difference
        # Возвращаем изменение весов
        return weight_difference

        #TODO: implement updating weights function
        #raise NotImplementedError('MomentumDescent update_weights function not implemented')


class Adam(VanillaGradientDescent):
    """
    Класс градиентного спуска с адаптивной оценкой моментов (Adam).

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    eps : float
        Малая добавка для предотвращения деления на ноль.
    m : np.ndarray
        Векторы первого момента.
    v : np.ndarray
        Векторы второго момента.
    beta_1 : float
        Коэффициент распада для первого момента.
    beta_2 : float
        Коэффициент распада для второго момента.
    iteration : int
        Счетчик итераций.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием адаптивной оценки моментов.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация класса Adam.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием адаптивной оценки моментов.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        self.iteration += 1
        # Обновление первого момента (среднее)
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        # Обновление второго момента (несмещенная оценка дисперсии)
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        
        # Коррекция смещения первого момента
        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        # Коррекция смещения второго момента
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)
        
        # Шаг оптимизации
        eta_k = self.lr.lambda_  # Вычисление длины шага на текущей итерации
        weight_diff = -eta_k * (m_hat / (np.sqrt(v_hat) + self.eps))
        self.w += weight_diff

        return weight_diff
        #raise NotImplementedError('Adagrad update_weights function not implemented')


class BaseDescentReg(BaseDescent):
    """
    Базовый класс для градиентного спуска с регуляризацией.

    Параметры
    ----------
    *args : tuple
        Аргументы, передаваемые в базовый класс.
    mu : float, optional
        Коэффициент регуляризации. По умолчанию равен 0.
    **kwargs : dict
        Ключевые аргументы, передаваемые в базовый класс.

    Атрибуты
    ----------
    mu : float
        Коэффициент регуляризации.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь с учетом L2 регуляризации по весам.
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        Инициализация базового класса для градиентного спуска с регуляризацией.
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь и L2 регуляризации по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом L2 регуляризации по весам.
        """
    
        """
        Вычисление градиента функции потерь и L2 регуляризации по весам.
        """
        # Вызов calc_gradient из базового класса для вычисления градиента функции потерь
        gradient = super().calc_gradient(x, y)
    
        # Добавление градиента L2 регуляризации к градиенту функции потерь
        # Учитываем, что для свободного члена (bias) регуляризация не применяется
        if hasattr(self, 'w'):
            l2_gradient = self.mu * self.w
        # Предполагаем, что self.w[0] это свободный член (bias), для которого регуляризация не применяется
            l2_gradient[0] = 0  
            gradient += l2_gradient
    
        return gradient

class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Класс полного градиентного спуска с регуляризацией.
    """
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Вызов calc_gradient из базового класса для вычисления градиента функции потерь
        gradient = super().calc_gradient(x, y)
    
        # Добавление градиента L2 регуляризации к градиенту функции потерь
        # Учитываем, что для свободного члена (bias) регуляризация не применяется
        if hasattr(self, 'w'):
            l2_gradient = self.mu * self.w
        # Предполагаем, что self.w[0] это свободный член (bias), для которого регуляризация не применяется
            l2_gradient[0] = 0  
            gradient += l2_gradient
    
        return gradient

class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Класс стохастического градиентного спуска с регуляризацией.
    """
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Вызов calc_gradient из базового класса для вычисления градиента функции потерь
        gradient = super().calc_gradient(x, y)
    
        # Добавление градиента L2 регуляризации к градиенту функции потерь
        # Учитываем, что для свободного члена (bias) регуляризация не применяется
        l2_gradient = self.mu * self.w
        # Предполагаем, что self.w[0] это свободный член (bias), для которого регуляризация не применяется
        l2_gradient[0] = 0  
        gradient += l2_gradient
    
        return gradient

class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Класс градиентного спуска с моментом и регуляризацией.
    """
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Вызов calc_gradient из базового класса для вычисления градиента функции потерь
        gradient = super().calc_gradient(x, y)
    
        # Добавление градиента L2 регуляризации к градиенту функции потерь
        # Учитываем, что для свободного члена (bias) регуляризация не применяется
        l2_gradient = self.mu * self.w
        # Предполагаем, что self.w[0] это свободный член (bias), для которого регуляризация не применяется
        l2_gradient[0] = 0  
        gradient += l2_gradient
    
        return gradient

class AdamReg(BaseDescentReg, Adam):
    """
    Класс адаптивного градиентного алгоритма с регуляризацией (AdamReg).
    """
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Вызов calc_gradient из базового класса для вычисления градиента функции потерь
        gradient = super().calc_gradient(x, y)
    
        # Добавление градиента L2 регуляризации к градиенту функции потерь
        # Учитываем, что для свободного члена (bias) регуляризация не применяется
        l2_gradient = self.mu * self.w
        # Предполагаем, что self.w[0] это свободный член (bias), для которого регуляризация не применяется
        l2_gradient[0] = 0  
        gradient += l2_gradient
    
        return gradient

def get_descent(descent_config: dict) -> BaseDescent:
    """
    Создает экземпляр класса градиентного спуска на основе предоставленной конфигурации.

    Параметры
    ----------
    descent_config : dict
        Словарь конфигурации для выбора и настройки класса градиентного спуска. Должен содержать ключи:
        - 'descent_name': строка, название метода спуска ('full', 'stochastic', 'momentum', 'adam').
        - 'regularized': булево значение, указывает на необходимость использования регуляризации.
        - 'kwargs': словарь дополнительных аргументов, передаваемых в конструктор класса спуска.

    Возвращает
    -------
    BaseDescent
        Экземпляр класса, реализующего выбранный метод градиентного спуска.

    Исключения
    ----------
    ValueError
        Вызывается, если указано неправильное имя метода спуска.

    Примеры
    --------
    >>> descent_config = {
    ...     'descent_name': 'full',
    ...     'regularized': True,
    ...     'kwargs': {'dimension': 10, 'lambda_': 0.01, 'mu': 0.1}
    ... }
    >>> descent = get_descent(descent_config)
    >>> isinstance(descent, BaseDescent)
    True
    """
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
