import logging
import grpc
from gen import kmeans_pb2 as pb2
from gen import kmeans_pb2_grpc as pb2_grpc
import pickle
import time
import itertools


def run():
    pks = [
        'msmarco_passage_01_796213134',
        'msmarco_passage_01_796213135',
        'msmarco_passage_01_796213136',
        'msmarco_passage_01_796213137',
        'msmarco_passage_01_796213138'
    ]

    vectors = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    ]


    with grpc.insecure_channel("localhost:50051") as channel:
        def request_iter():
            for pk, vector in zip(pks, vectors):
                print(pk)
                print(vector)
                yield pb2.PkVector(pk=pk, vector=vector)

        stub = pb2_grpc.KmeansStub(channel)
        resp = stub.PutPkVector(request_iter(), timeout=60)
        print(resp)


if __name__ == '__main__':
    logging.basicConfig()
    for i in range(1000):
        run()
        # time.sleep(3)
