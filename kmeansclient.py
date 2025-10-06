import logging
import grpc
from gen import kmeans_pb2 as pb2
from gen import kmeans_pb2_grpc as pb2_grpc
import pickle


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

    batch = pb2.VectorBatch(
        data = pickle.dumps((pks, vectors), protocol=pickle.HIGHEST_PROTOCOL)
    )
    print(batch)


    with grpc.insecure_channel("localhost:50051") as channel:
        stub = pb2_grpc.KmeansStub(channel)
        future = stub.PutVectorBatch.future(batch)
        result = future.result()
        print(result)


if __name__ == '__main__':
    logging.basicConfig()
    run()
    