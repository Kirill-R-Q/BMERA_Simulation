"""PennyLane circuit reference for BMERA."""

import pennylane as qml
from pennylane import numpy as np

def BMERA_CRZ_block(params, wires):
    """
    Применяет квантовую схему, показанную на диаграмме, к двум указанным кубитам.

    Схема состоит из:
    1. Параллельные гейты RX на обоих кубитах.
    2. Параллельные гейты RZ на обоих кубитах.
    3. Гейт CRZ, где управляющий кубит - нижний, а целевой - верхний.

    Args:
        params (list[float]): Список из 5 параметров для гейтов вращения:
                              [theta_rx_q0, theta_rx_q1, theta_rz_q0, theta_rz_q1, theta_crz]
                              - theta_rx_q0: Параметр для RX на верхнем кубите.
                              - theta_rx_q1: Параметр для RX на нижнем кубите.
                              - theta_rz_q0: Параметр для RZ на верхнем кубите (перед CRZ).
                              - theta_rz_q1: Параметр для RZ на нижнем кубите.
                              - theta_crz:   Параметр для гейта CRZ.
        wires (list[int]): Список из двух индексов кубитов [q_top, q_bottom],
                           где q_top соответствует верхней линии на диаграмме,
                           а q_bottom - нижней.
    """
    if params.size != 5:
        raise ValueError(f"Ожидается 5 параметров для гейтов: [RX_q0, RX_q1, RZ_q0, RZ_q1, CRZ], {params.size}")
    if len(wires) != 2:
        raise ValueError("Ожидается 2 индекса кубитов: [q_top, q_bottom].")

    q_top, q_bottom = wires[0], wires[1]
    theta_rx_q0, theta_rx_q1, theta_rz_q0, theta_rz_q1, theta_crz = params

    # Слой 1: Параллельные гейты RX
    qml.RX(theta_rx_q0, wires=q_top)
    qml.RX(theta_rx_q1, wires=q_bottom)

    # Слой 2: Параллельные гейты RZ
    qml.RZ(theta_rz_q0, wires=q_top)
    qml.RZ(theta_rz_q1, wires=q_bottom)

    # Слой 3: Управляемый гейт RZ
    # Управляющий кубит: q_bottom (нижняя линия)
    # Целевой кубит: q_top (верхняя линия)
    qml.CRZ(theta_crz, wires=[q_bottom, q_top])

def BMERA_CRX_block(params, wires):
    if params.size != 5:
        raise ValueError(f"Ожидается 5 параметров для гейтов: [RX_q0, RX_q1, RZ_q0, RZ_q1, CRX], {params.size}")
    if len(wires) != 2:
        raise ValueError("Ожидается 2 индекса кубитов: [q_top, q_bottom].")

    q_top, q_bottom = wires[0], wires[1]
    theta_rx_q0, theta_rx_q1, theta_rz_q0, theta_rz_q1, theta_crx = params

    # Слой 1: Параллельные гейты RX
    qml.RX(theta_rx_q0, wires=q_top)
    qml.RX(theta_rx_q1, wires=q_bottom)

    # Слой 2: Параллельные гейты RZ
    qml.RZ(theta_rz_q0, wires=q_top)
    qml.RZ(theta_rz_q1, wires=q_bottom)

    # Слой 3: Управляемый гейт RX
    # Управляющий кубит: q_bottom (нижняя линия)
    # Целевой кубит: q_top (верхняя линия)
    qml.CRX(theta_crx, wires=[q_bottom, q_top])

def BMERA_CNOT_block(params, wires):
    q_top, q_bottom = wires[0], wires[1]
    theta_rx_q0, theta_rx_q1, theta_rz_q0, theta_rz_q1 = params

    # Слой 1: Параллельные гейты RX
    qml.RX(theta_rx_q0, wires=q_top)
    qml.RX(theta_rx_q1, wires=q_bottom)

    # Слой 2: Параллельные гейты RZ
    qml.RZ(theta_rz_q0, wires=q_top)
    qml.RZ(theta_rz_q1, wires=q_bottom)

    qml.CNOT(wires=[q_bottom, q_top])

def RXX(theta):
    rxx = np.array([
        [np.cos(theta/2), 0, 0, -1j*np.sin(theta/2)],
        [0, np.cos(theta/2), -1j*np.sin(theta/2), 0],
        [0, -1j*np.sin(theta/2), np.cos(theta/2), 0],
        [-1j*np.sin(theta/2), 0, 0, np.cos(theta/2)]
    ])
    return rxx

def BMERA_RXX_block(params, wires):
    if params.size != 5:
        raise ValueError(f"Ожидается 5 параметров для гейтов: [RX_q0, RX_q1, RZ_q0, RZ_q1, RXX], {params.size}")
    if len(wires) != 2:
        raise ValueError("Ожидается 2 индекса кубитов: [q_top, q_bottom].")

    q_top, q_bottom = wires[0], wires[1]
    theta_rx_q0, theta_rx_q1, theta_rz_q0, theta_rz_q1, theta_rxx = params

    # Слой 1: Параллельные гейты RX
    qml.RX(theta_rx_q0, wires=q_top)
    qml.RX(theta_rx_q1, wires=q_bottom)

    # Слой 2: Параллельные гейты RZ
    qml.RZ(theta_rz_q0, wires=q_top)
    qml.RZ(theta_rz_q1, wires=q_bottom)

    # Слой 3: Управляемый гейт RZ
    # Управляющий кубит: q_bottom (нижняя линия)
    # Целевой кубит: q_top (верхняя линия)
    qml.QubitUnitary(RXX(theta_rxx), wires=[q_bottom, q_top])


def bmera_swap_layer(n_total: int, n: int, start_qubit = 0):
    """
    Применяет операцию перестановки (swap) для 2n кубитов,
    используемую в анзатце BMERA.

    Правила перестановки:
    - Всего кубитов: 2n.
    - Кубиты от 0 до n-2 переходят в 2*(i+1).
    - Кубит n-1 переходит в 0.
    - Кубиты от n до 2n-1 (представленные как n+j) переходят в 2*j + 1.

    Args:
        n (int): Половина от общего числа кубитов.
        start_qubit (int): первый кубит в группе (default 0)
    """
    num_qubits = 2 * n
    if num_qubits == 0:
        return

    # --- Шаг 1: Создаем карту целевой перестановки ---
    # target_map[i] = j означает, что кубит, который был на позиции i, должен оказаться на позиции j.
    target_map = {}
    """
    # Правило для кубитов от 0 до n-2
    for i in range(start_qubit, n - 1 + start_qubit):
        target_map[i] = 2 * (i + 1) - start_qubit

    # Особое правило для кубита n-1
    target_map[n - 1 + start_qubit] = start_qubit

    # Правило для кубитов от n до 2n-1
    for j in range(start_qubit, n + start_qubit):
        # i = n + j
        target_map[n + j] = 2 * j + 1 - start_qubit
    """
    for i in range(n_total):
        if (start_qubit <= i < n-1+start_qubit):
            target_map[i] = 2 * (i + 1) - start_qubit
        elif (i == n - 1 + start_qubit):
            target_map[i] = start_qubit
        elif (2*n - 1 + start_qubit >= i > n - 1 + start_qubit):
            target_map[i] = 2 * (i - n) + 1 - start_qubit
        else:
            target_map[i] = i


    # --- Шаг 2: Генерируем SWAP-гейты для достижения перестановки ---
    # Мы будем отслеживать, где находится каждый *логический* кубит.
    # current_pos[logical_idx] = physical_idx
    # Изначально логический кубит i находится на физической позиции i.
    current_pos = list(range(n_total))

    # Чтобы найти, какой логический кубит находится на физической позиции,
    # нам нужен обратный маппинг.
    # physical_pos_to_logical[physical_idx] = logical_idx
    physical_pos_to_logical = list(range(n_total))

    # Проходим по каждой физической позиции и ставим на нее правильный кубит.
    for physical_wire in range(n_total):
        # Какой логический кубит должен быть на этой позиции в итоге?
        # Мы ищем `k` такой, что `target_map[k] = physical_wire`.
        # Это инверсия target_map.
        logical_qubit_that_should_be_here = -1
        for k, v in target_map.items():
            if v == physical_wire:
                logical_qubit_that_should_be_here = k
                break

        # Где этот нужный нам кубит находится сейчас?
        current_physical_pos_of_target = current_pos[logical_qubit_that_should_be_here]

        # Если он уже на своем месте, ничего не делаем.
        if current_physical_pos_of_target == physical_wire:
            continue

        # Если нет, меняем его местами с тем, что сейчас на `physical_wire`.
        qml.SWAP(wires=[physical_wire, current_physical_pos_of_target])

        # --- Обновляем наши отслеживающие списки после SWAP ---
        # Какой логический кубит был на `physical_wire` до SWAP?
        logical_qubit_that_was_here = physical_pos_to_logical[physical_wire]

        # Теперь их физические позиции поменялись местами.
        current_pos[logical_qubit_that_should_be_here] = physical_wire
        current_pos[logical_qubit_that_was_here] = current_physical_pos_of_target

        # Обновляем и обратный маппинг.
        physical_pos_to_logical[physical_wire] = logical_qubit_that_should_be_here
        physical_pos_to_logical[current_physical_pos_of_target] = logical_qubit_that_was_here


def pennylane_bmera_expvals(in_data, n_qubits, type_, params):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(in_data, params):
        qml.AngleEmbedding(features=2 * in_data, wires=range(n_qubits), rotation='Y')
        d = int(np.log2(n_qubits))
        width = n_qubits // 2
        network_params = params.reshape((d, width * 2, params.shape[-1]))

        if type_ == 1:
            block = BMERA_CNOT_block
        elif type_ == 2:
            block = BMERA_CRX_block
        elif type_ == 3:
            block = BMERA_CRZ_block
        else:
            raise ValueError('type can be only 1..3')

        for j in range(1, d + 1):
            for start_qubit in range(0, n_qubits, 2**j):
                bmera_swap_layer(n_qubits, 2**(j - 1), start_qubit)

            layer_params = network_params[j - 1]
            if j == 1:
                for start_qubit in range(0, n_qubits, 2):
                    block(params=layer_params[start_qubit + 1], wires=[start_qubit, start_qubit + 1])
            else:
                for start_qubit in range(0, n_qubits, 2**j):
                    for i in range(2**j // 2):
                        block(params=layer_params[2 * i + start_qubit], wires=[2 * i + start_qubit, 2 * i + 1 + start_qubit])
                    for i in range(2**j - 1):
                        qml.SWAP(wires=[i + start_qubit, i + 1 + start_qubit])
                    for i in range(2**j // 2):
                        block(params=layer_params[start_qubit + 2 * i + 1], wires=[2 * i + start_qubit, 2 * i + 1 + start_qubit])

        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return np.array(circuit(in_data, params))
