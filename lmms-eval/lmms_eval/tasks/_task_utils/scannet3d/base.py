from __future__ import annotations

import json
from typing import Callable

import datasets
from PIL import Image

from lmms_eval.api.task import ConfigurableTask

from .data import get_asset_manager


def open_frame_images(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    assets = get_asset_manager()
    return assets.open_images(doc["frame_files"])


VIDEO3D_EXTRA_PROMPT = (
    "The video captures 3D spatial information of a scene. "
    "Please focus on the spatial relationships in the video and answer the following questions.\n"
)


def prepend_video3d_extra_prompt(text: str) -> str:
    return f"{VIDEO3D_EXTRA_PROMPT}{text}"


class Scannet3DConfigurableTask(ConfigurableTask):
    BENCHMARK_NAME = None
    DEFAULT_SPLIT = "val"
    DEFAULT_NUM_FRAMES = 32
    DEFAULT_SAMPLING = "uniform"

    def __init__(self, config=None, model_name=None):
        merged = {
            "output_type": "generate_until",
            "doc_to_visual": open_frame_images,
            "num_fewshot": 0,
            "generation_kwargs": {"max_new_tokens": 128, "temperature": 0.0},
            "test_split": self.DEFAULT_SPLIT,
            "metadata": {
                "version": 1,
                "benchmark_name": self.BENCHMARK_NAME,
                "num_frames": self.DEFAULT_NUM_FRAMES,
                "sampling_strategy": self.DEFAULT_SAMPLING,
            },
        }
        if config:
            merged.update(config)
        super().__init__(config=merged, model_name=model_name)

    def download(self, dataset_kwargs=None) -> None:
        del dataset_kwargs
        docs = self.build_docs(
            split=self.config.test_split or self.config.validation_split or self.DEFAULT_SPLIT,
            max_frames=self.config.metadata.get("num_frames", self.DEFAULT_NUM_FRAMES),
            strategy=self.config.metadata.get("sampling_strategy", self.DEFAULT_SAMPLING),
        )
        split_name = self.config.test_split or self.config.validation_split or self.DEFAULT_SPLIT
        dataset = datasets.Dataset.from_list(docs)
        self.dataset = datasets.DatasetDict({split_name: dataset})
        self.dataset_no_image = self.dataset

    def build_docs(self, split: str, max_frames: int, strategy: str):
        raise NotImplementedError
