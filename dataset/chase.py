from dataset.dataset_wo_masks import DatasetWoMasks
from network.chase import ChaseNetwork

class ChaseDataset(DatasetWoMasks):

    def __init__(self, batch_size=1, WRK_DIR_PATH ="./chase", TRAIN_SUBDIR="train", TEST_SUBDIR="test", sgd = True,
                 cv_train_inds = None, cv_test_inds = None):
        super(ChaseDataset, self).__init__(batch_size=batch_size, WRK_DIR_PATH=WRK_DIR_PATH, TRAIN_SUBDIR=TRAIN_SUBDIR,
                                         TEST_SUBDIR=TEST_SUBDIR, sgd=sgd, cv_train_inds=cv_train_inds,
                                         cv_test_inds=cv_test_inds)

    @property
    def network_cls(self):
        return ChaseNetwork
