import os

import PIL.Image as Image
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class HMDB51Dataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split  # 'train', 'val', 'test'

        # HMDB51의 비디오 경로와 레이블을 로드합니다.
        self.video_paths, self.labels = self._load_annotations()

    def _load_annotations(self):
        video_paths = []
        labels = []

        split_dir = self.data_dir + '_split'  # split 파일이 위치한 디렉토리 경로
        split_number = 1  # 사용할 split 번호 (1, 2, 3 중 선택)

        for class_idx, class_name in enumerate(sorted(os.listdir(self.data_dir))):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # 해당 클래스의 split 파일 열기
            split_file = os.path.join(split_dir, f'{class_name}_test_split{split_number}.txt')
            with open(split_file, 'r') as f:
                for line in f:
                    video_name, split_label = line.strip().split()
                    video_name = video_name.replace('.avi', '')
                    split_label = int(split_label)
                    if self.split == 'train' and split_label == 1:
                        video_paths.append(os.path.join(class_dir, video_name))
                        labels.append(class_idx)
                    elif self.split == 'test' and split_label == 2:
                        video_paths.append(os.path.join(class_dir, video_name))
                        labels.append(class_idx)
                    elif self.split == 'val' and split_label == 0:
                        video_paths.append(os.path.join(class_dir, video_name))
                        labels.append(class_idx)

        return video_paths, labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._load_video_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return frames, label

    def _load_video_frames(self, video_path):
        frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg') or f.endswith('.png')]
        frame_files.sort()  # 프레임 순서를 맞추기 위해 정렬

        frames = []
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(video_path, frame_file)
            frame = Image.open(frame_path).convert('RGB')  # 이미지를 RGB로 로드
            frames.append(frame)

        return frames

class HMDB51DataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=2, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),       # 먼저 256x256으로 리사이즈
            transforms.ToTensor(),               # 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = HMDB51Dataset(self.data_dir, split='train', transform=self.transform)
            self.val_dataset = HMDB51Dataset(self.data_dir, split='val', transform=self.transform)
        if stage == 'test' or stage is None:
            self.test_dataset = HMDB51Dataset(self.data_dir, split='test', transform=self.transform)

    def prepare_data(self):
        pass

    def collate_fn(self, batch):
        frames_list, labels = zip(*batch)  # 배치에서 프레임과 레이블 분리

        # 각 비디오의 시퀀스 길이 얻기
        seq_lengths = [len(frames) for frames in frames_list]
        max_seq_len = max(seq_lengths)

        # 패딩된 시퀀스를 저장할 리스트
        padded_frames = []

        for frames in frames_list:
            seq_len = len(frames)
            if seq_len < max_seq_len:
                # 패딩 추가
                num_pads = max_seq_len - seq_len
                pad_frame = torch.zeros_like(frames[0])  # 제로 텐서 생성
                frames.extend([pad_frame] * num_pads)
            padded_frames.append(torch.stack(frames))  # (seq_len, C, H, W)

        # 배치 차원을 추가하여 텐서로 변환
        batch_frames = torch.stack(padded_frames)  # (batch_size, seq_len, C, H, W)
        labels = torch.tensor(labels)

        return batch_frames, labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
                          , collate_fn=self.collate_fn, persistent_workers=True, multiprocessing_context='fork' if torch.backends.mps.is_available() else None)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
                          , persistent_workers=True, multiprocessing_context='fork' if torch.backends.mps.is_available() else None)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
                          , persistent_workers=True, multiprocessing_context='fork' if torch.backends.mps.is_available() else None)
