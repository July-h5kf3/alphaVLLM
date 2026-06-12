from collections import deque
class Block:
    def __init__(
        self,
        block_id,
    ):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
    def update(
        self,
        hash: int,
        token_ids: list[int],
    ):
        self.hash = hash
        self.token_ids = token_ids
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class BlockManager:

    def __init__(self, num_blocks: int, block_size : int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()