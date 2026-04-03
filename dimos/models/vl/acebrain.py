from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from openai import OpenAI

from dimos.models.vl.base import VlModel, VlModelConfig
from dimos.msgs.sensor_msgs import Image


@dataclass
class AceBrainVlModelConfig(VlModelConfig):
    model_name: str = "/data/users/ace_brain_8b"
    base_url: str = "http://103.237.28.254:8080/v1"
    api_key: str = "EMPTY"


class AceBrainVlModel(VlModel):
    default_config = AceBrainVlModelConfig
    config: AceBrainVlModelConfig

    @cached_property
    def _client(self) -> OpenAI:
        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    def query(  # type: ignore[override]
        self,
        image: Image | np.ndarray,
        query: str,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(image, np.ndarray):
            image = Image.from_numpy(image)

        image, _ = self._prepare_image(image)
        img_base64 = image.to_base64()

        api_kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ],
        }
        if response_format:
            api_kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**api_kwargs)
        return response.choices[0].message.content or ""

    def query_text(
        self,
        query: str,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        api_kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": query}],
        }
        if response_format:
            api_kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**api_kwargs)
        return response.choices[0].message.content or ""

    def stop(self) -> None:
        if "_client" in self.__dict__:
            del self.__dict__["_client"]
