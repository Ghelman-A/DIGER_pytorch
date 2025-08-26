from torch.utils.data import Sampler
import pickle
from typing import Optional, Sized, List, Iterator, TypeVar

T_co = TypeVar('T_co', covariant=True)


class PredictSampler(Sampler):
    """
        The aim of this class is to sample only a subset of dataset videos during prediction. For instance, if we
    require the model output for only one or a few videos, their names are passed into the sampler as a list and
    the prediction dataloader will only return those videos. Otherwise, the entire videos in the dataset will pass
    through the model for prediction.

    """
    def __init__(self, data_source: Optional[Sized], pkl_file: str, pred_list: List[str]=None) -> None:
        super().__init__(data_source)

        self.pred_list = pred_list
        with open(pkl_file, 'rb') as f:
            dataset_df = pickle.load(f)
        
        self.pred_data = []
        if (pred_list is not None) and (len(pred_list) > 0):
            dset_list = ['_'.join(data[0].name.split('_')[:2]) for data in dataset_df['frame_names']]
            self.pred_data = [dset_list.index(vid) for vid in pred_list]
        else:
            self.pred_data = list(range(len(dataset_df)))

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.pred_data)

    def __len__(self):
        return len(self.pred_data)