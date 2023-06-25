import paddle.distributed.fleet as fleet
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class SingleTaskReader(fleet.MultiSlotDataGenerator):
    def init(self):
        padding = 0
        sparse_slots = "log_key click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = ["dense_feature"]
        self.dense_slots_shape = [13]
        self.slots = self.sparse_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding
        logger.info("pipe init success")

    def line_process(self, line):
        items = line.strip("\n").split("\t")
        log_key = int(items[0])
        conv1 = items[1]
        conv2 = items[2]
        conv3 = items[3]
        conv = 0 
        if conv1 == "1" or conv2 == "1" or conv3 == "1":
            conv = 1
        else:
            conv = 0
        output = [(i, []) for i in self.slots]
        feasigns = items[4].split(" ")
        for i in feasigns:
            slot_feasign = i.split(":")
            slot = slot_feasign[1]
            if slot not in self.slots:
                continue
            if slot in self.sparse_slots:
                feasign = int(slot_feasign[0])
            else:
                feasign = float(slot_feasign[0])
            output[self.slot2index[slot]][1].append(feasign)
            self.visit[slot] = True
        output[0][1].append(log_key)
        self.visit['log_key'] = True
        output[1][1].append(conv)
        self.visit['click'] = True
        for i in self.visit:
            slot = i
            if not self.visit[slot]:
                if i in self.dense_slots:
                    output[self.slot2index[i]][1].extend(
                        [self.padding] *
                        self.dense_slots_shape[self.slot2index[i]])
                else:
                    output[self.slot2index[i]][1].extend([self.padding])
            else:
                self.visit[slot] = False

        return output
    
    def generate_sample(self, line):
        r"Dataset Generator"

        def reader():
            output_dict = self.line_process(line)
            yield output_dict

        return reader


if __name__ == "__main__":
    r = SingleTaskReader()
    r.init()
    r.run_from_stdin()
