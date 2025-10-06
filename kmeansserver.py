import asyncio
import grpc
from gen import kmeans_pb2 as pb2
from gen import kmeans_pb2_grpc as pb2_grpc
import numpy as np
import pickle


# async def worker(name, queue):
#     while True:
#         # Get a "work item" out of the queue.
#         sleep_for = await queue.get()

#         # Sleep for the "sleep_for" seconds.
#         # await asyncio.sleep(sleep_for)

#         # Notify the queue that the "work item" has been processed.
#         queue.task_done()

#         print(f'{name} has slept for {sleep_for:.2f} seconds')


# async def main():
#     # Create a queues to recieve vector to process
#     # and labels+centroids results
#     vectors_in_queue = asyncio.Queue()
#     labels_out_queue = asyncio.Queue()

#     # Generate random timings and put them into the queue.
#     # total_sleep_time = 0
#     # for _ in range(20):
#     #     sleep_for = random.uniform(0.05, 1.0)
#     #     total_sleep_time += sleep_for
#     #     queue.put_nowait(sleep_for)

#     # There is a single worker that runs kmeans partial fit
#     task = asyncio.create_task(worker(f"kmeans", vectors_in_queue))

#     # Wait until the queue is fully processed.
#     # started_at = time.monotonic()
#     await vectors_in_queue.join()
#     # total_slept_for = time.monotonic() - started_at

#     # Cancel our worker tasks.
#     task.cancel()
#     # Wait until all worker tasks are cancelled.
#     (result,) = await asyncio.gather(task, return_exceptions=True)

#     # print('====')
#     # print(f'3 workers slept in parallel for {total_slept_for:.2f} seconds')
#     # print(f'total expected sleep time: {total_sleep_time:.2f} seconds')


# asyncio.run(main())


verbose = True


class KmeansService(pb2_grpc.KmeansServicer):
    def __init__(self):
        self.vectors_in_queue = asyncio.Queue()
        self.labels_out_queue = asyncio.Queue()


    async def PutVectorBatch(self, request: pb2.VectorBatch, context):
        # do non-blocking work here (or offload CPU work; see note below)
        pks, vectors = pickle.loads(request.data)
        vectors = np.array(vectors, dtype=np.float32)

        # 3) single-batch update + labels
        labels = cluster_kmeans(model, all_vectors, verbose=verbose)
        centroids = model.cluster_centers_

        if verbose:
            print(f"[INFO] Labels: {labels}")
            print(f"[INFO] Centroids shape: {centroids.shape}")

        # # 4) persist in a single retriable txn
        # epoch = persist_batch(conn, table, column, all_pks, labels, centroids, verbose=verbose, dry_run=dry_run)

        # if verbose:
        #     print(f"[INFO] Completed iteration (epoch={epoch})")


        return pb2.VectorBatchAck(ok=True)





async def serve():
    server = grpc.aio.server()
    pb2_grpc.add_KmeansServicer_to_server(KmeansService(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    print("async gRPC server on 50051")
    await server.wait_for_termination()



if __name__ == "__main__":
    asyncio.run(serve())