## 数据格式

目录data/下包含train/dev/test.doc.json和train/dev.ann.json，分别为文档文件和注释文件。

文档文件（即.doc.json文件）包含标注前的原文档和事件短语，格式如下：

```json
{
    "Descriptor": {
        "event_id": event_id,
        "text": "事件短语"
    },
    "Doc": {
        "doc_id": doc_id,
        "title": "文档的标题",
        "content": [
            {
                "sent_idx": 0,
                "sent_text": "第1个句子文本。"
            },
            {
                "sent_idx": 1,
                "sent_text": "第2个句子文本。"
            },
            ...
            {
                "sent_idx": n-1,
                "sent_text": "第n个句子文本"
            }
        ]
    }
}
```

文档文件的每个JSON实例为一篇文档，包含`Descriptor`和`Doc`字段，其中`Descriptor`表示事件短语，`Doc`表示与事件相关的文档。

注释文件包含在文档文件基础上标注的观点片段及其观点目标，格式如下：

```json
[
	{
        "event_id": (int) event_id,
        "doc_id": (int) doc_id,
        "start_sent_idx": (int) "观点片段开始的句子位置",
        "end_sent_idx": (int) "观点片段结束的句子位置",
        "argument": (str) "观点片段对应的事件论元（观点目标）"
  	}
]
```

注释文件的每个JSON实例为一个观点片段，其中`event_id`字段为观点片段相关的事件id，`doc_id`为观点片段所在的文档id，`start_sent_idx`和`end_sent_idx`字段分别为观点片段开始和结束对应的句子id，`argument`字段为观点片段对应的观点目标。

## 提交格式

参赛队伍需提交**测试集**的预测结果。预测文件**格式**应与”data/“目录下的“train/dev.ann.json”相同，**命名**为”test.ann.json“。
