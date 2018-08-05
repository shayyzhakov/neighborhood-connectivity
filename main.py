import pandas as pd
import numpy as np
import threading
import math
from reedsolo import RSCodec, ReedSolomonError
import random

generate_input = True
n = 49  # can be any number that has a square root
parties = []
k = int(math.sqrt(n))
outputs = {}
barrier_count = 1
rs_correction_len = 20
p = 0.1


def barrier_end_action():
    global barrier_count
    print('---round {} ended---'.format(barrier_count))
    barrier_count += 1


round_barrier = threading.Barrier(n, action=barrier_end_action)
barrier = threading.Barrier(n)
last_barrier = threading.Barrier(n + 1, action=lambda: print('...'))


class PartyThread(threading.Thread):
    def __init__(self, i, input_vector):
        self.i = i
        self.input_vector = input_vector
        self.output_vector = [-1] * n
        self.minor_matrix = np.full((k, k), -1)
        self.encoded_minor_matrix = np.full((k, k + rs_correction_len), -1)
        self.encoded_output_vectors = np.full((k, k + rs_correction_len), -1)
        self.round = 0
        self.B = (i // k, i % k)
        self.buffer = {x: [] for x in range(n)}
        threading.Thread.__init__(self)

    def run(self):
        # fill known row in matrix
        self.minor_matrix[self.B[1]] = self.input_vector[(self.B[1] * k):(self.B[1] * k + k)]

        # wait for other threads to be ready for the first round
        round_barrier.wait()
        rs = RSCodec(rs_correction_len)
        vectors_to_send = [rs.encode(self.input_vector[i*k:i*k+k].tolist()) for i in range(k)]

        # O(num_of_rounds) = O(1)
        num_of_rounds = math.ceil((k + rs_correction_len) / k)

        # send bit a(i,j) to party ((ki + j)%n)
        for r in range(num_of_rounds):
            for block in range(k):
                for j in range(k):
                    if len(vectors_to_send[block]) > r * k + j:
                        middle_man = (k * self.i + (k * block + j)) % n
                        msg = vectors_to_send[block][r * k + j]
                        self.send_buffer_message(middle_man, msg)

            # wait fo other threads to be ready for another round
            round_barrier.wait()

        # send to final target (self.i == middleman)
        for r in range(num_of_rounds):
            for src in self.buffer:
                if len(self.buffer[src]) > 0:
                    j = (n - ((src * int(math.sqrt(n))) % n) + self.i) % n
                    t = src // k
                    l = j // k
                    target = k*t + l
                    self.send_minor_matrix_message(target, self.buffer[src].pop(0))

            # wait fo other threads to be ready for another round
            round_barrier.wait()
            self.round += 1
            barrier.wait()

        # decode messages
        for x in range(k):
            try:
                self.minor_matrix[x] = rs.decode(self.encoded_minor_matrix[x])
            except ReedSolomonError:
                print('Failed to decode! terminating...')
                self.minor_matrix[x] = [2] * k

        # wait for other threads to fill the minor matrix
        round_barrier.wait()
        self.round = 0

        vectors_to_send = [rs.encode(list(self.minor_matrix[:, x])) for x in range(k)]

        # send bit a(i,j) to party ((i + kj)%n) second time
        for r in range(num_of_rounds):
            for i in range(k):
                for j in range(k):
                    if len(vectors_to_send[j]) > r * k + i:
                        real_i = self.B[0] * k + i
                        real_j = self.B[1] * k + j
                        middle_man = (real_i + k * real_j) % n
                        msg = vectors_to_send[j][r * k + i]
                        self.send_buffer_message(middle_man, msg)

            # wait fo other threads to be ready for another round
            round_barrier.wait()

        # wait for other threads to be ready for the second round the second time
        round_barrier.wait()

        # send to final target (self.i == middleman) second time
        for r in range(num_of_rounds):
            for src in self.buffer:
                if len(self.buffer[src]) > 0:
                    for i in range(k):
                        for j in range(k):
                            real_i = parties[src].B[0] * k + i
                            real_j = parties[src].B[1] * k + j
                            if self.i == (real_i + k * real_j) % n:
                                self.send_output_message(real_j, self.buffer[src].pop(0))

            # wait fo other threads to be ready for another round
            round_barrier.wait()
            self.round += 1
            barrier.wait()

        # decode messages
        for x in range(k):
            try:
                self.output_vector[x*k:x*k+k] = rs.decode(self.encoded_output_vectors[x])
            except ReedSolomonError:
                print('Failed to decode!')
                self.output_vector[x * k:x * k + k] = [2] * k

        # wait for all threads to finish
        round_barrier.wait()

        # write to output matrix
        outputs[self.i] = self.output_vector

        # done everything
        last_barrier.wait()

    def send_buffer_message(self, destination, msg):
        if random.random() < p:
            msg = 1 - msg
        parties[destination].receive_buffer_message(self.i, msg)

    def receive_buffer_message(self, source, msg):
        self.buffer[source].append(msg)

    def send_minor_matrix_message(self, destination, msg):
        if random.random() < p:
            msg = 1 - msg
        parties[destination].receive_minor_matrix_message(self.i, msg)

    def receive_minor_matrix_message(self, source, msg):
        for i in range(n):
            for j in range(n):
                if (self.i == k * (i // k) + (j // k)) and ((k * i + j) % n == source):
                    self.encoded_minor_matrix[i % k][self.round * k + j % k] = msg

    def send_output_message(self, destination, msg):
        if random.random() < p:
            msg = 1 - msg
        parties[destination].receive_output_message(self.i, msg)

    def receive_output_message(self, source, msg):
        i = (n - ((self.i * int(math.sqrt(n))) % n) + source) % n
        self.encoded_output_vectors[i // k][self.round * k + i % k] = msg


def main():
    if generate_input is True:
        input_matrix = pd.DataFrame(np.random.randint(0, 2, size=(n, n)))
    else: # read input matrix from file
        input_matrix = pd.read_excel("input.xlsx", header=None)

    matrix = input_matrix.values

    # run n different threads
    for i in range(n):
        input_vector = matrix[i]
        t = PartyThread(i, input_vector)
        parties.append(t)
        t.start()

    last_barrier.wait()
    output_matrix = pd.DataFrame(outputs, columns=range(n))

    # write to file
    writer = pd.ExcelWriter('matrix.xlsx')
    input_matrix.to_excel(writer, 'InputSheet')
    output_matrix.to_excel(writer, 'OutputSheet')
    writer.save()

    # checks validity by calculating the difference between the input matrix and the output matrix
    diff_matrix = input_matrix - output_matrix
    if not diff_matrix.any().any():  # .any().any() checks if there exist any non-zero value
        print('Success!')
    else:
        print('Failed...')


if __name__ == '__main__':
    main()
