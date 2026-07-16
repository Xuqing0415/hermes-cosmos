import json
import time
from typing import List, Dict, Optional, Any
from enum import Enum

from .types import Message, MessageType, AgentInfo, Task


class CommunicationChannel:
    def __init__(self):
        self.messages: List[Message] = []
        self.subscriptions: Dict[str, List[str]] = {}

    def send(self, message: Message):
        self.messages.append(message)

    def receive(self, receiver_id: str) -> List[Message]:
        messages = [m for m in self.messages 
                   if m.receiver_id is None or m.receiver_id == receiver_id]
        return messages

    def subscribe(self, agent_id: str, message_type: MessageType):
        key = message_type.value
        if key not in self.subscriptions:
            self.subscriptions[key] = []
        if agent_id not in self.subscriptions[key]:
            self.subscriptions[key].append(agent_id)

    def unsubscribe(self, agent_id: str, message_type: MessageType):
        key = message_type.value
        if key in self.subscriptions and agent_id in self.subscriptions[key]:
            self.subscriptions[key].remove(agent_id)

    def broadcast(self, message: Message):
        message.receiver_id = None
        self.messages.append(message)

    def get_subscribers(self, message_type: MessageType) -> List[str]:
        return self.subscriptions.get(message_type.value, [])

    def clear_messages(self):
        self.messages.clear()


class AgentProtocol:
    def __init__(self, channel: Optional[CommunicationChannel] = None):
        self.channel = channel or CommunicationChannel()

    def send_message(self, sender_id: str, message_type: MessageType,
                     content: Dict[str, Any], receiver_id: Optional[str] = None,
                     task_id: Optional[str] = None) -> Message:
        message = Message(
            message_id="",
            type=message_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            task_id=task_id,
            content=content
        )
        self.channel.send(message)
        return message

    def send_task_publish(self, sender_id: str, task: Task) -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.TASK_PUBLISH,
            content={
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "type": task.type,
                "input_data": task.input_data,
                "priority": task.priority,
                "deadline": task.deadline
            }
        )

    def send_task_bid(self, sender_id: str, task_id: str, 
                      bid_details: Dict[str, Any]) -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.TASK_BID,
            content={
                "task_id": task_id,
                "bidder_id": sender_id,
                **bid_details
            }
        )

    def send_task_assign(self, sender_id: str, task_id: str, 
                         assignee_id: str) -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.TASK_ASSIGN,
            content={
                "task_id": task_id,
                "assignee_id": assignee_id
            },
            receiver_id=assignee_id
        )

    def send_task_update(self, sender_id: str, task_id: str,
                         update_content: Dict[str, Any]) -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.TASK_UPDATE,
            content={
                "task_id": task_id,
                **update_content
            }
        )

    def send_task_complete(self, sender_id: str, task_id: str,
                           result: Dict[str, Any]) -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.TASK_COMPLETE,
            content={
                "task_id": task_id,
                "result": result
            }
        )

    def send_knowledge_share(self, sender_id: str, knowledge: Dict[str, Any]) -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.KNOWLEDGE_SHARE,
            content=knowledge
        )

    def send_knowledge_request(self, sender_id: str, query: Dict[str, Any]) -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.KNOWLEDGE_REQUEST,
            content=query
        )

    def send_heartbeat(self, sender_id: str, status: str = "active") -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.HEARTBEAT,
            content={"status": status}
        )

    def send_register(self, sender_id: str, agent_info: AgentInfo) -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.REGISTER,
            content={
                "agent_id": agent_info.agent_id,
                "role": agent_info.role.value,
                "name": agent_info.name,
                "capabilities": agent_info.capabilities
            }
        )

    def send_deregister(self, sender_id: str) -> Message:
        return self.send_message(
            sender_id=sender_id,
            message_type=MessageType.DEREGISTER,
            content={"agent_id": sender_id}
        )

    def receive_messages(self, agent_id: str) -> List[Message]:
        return self.channel.receive(agent_id)

    def broadcast_message(self, sender_id: str, message_type: MessageType,
                         content: Dict[str, Any]) -> Message:
        message = Message(
            message_id="",
            type=message_type,
            sender_id=sender_id,
            content=content
        )
        self.channel.broadcast(message)
        return message

    def subscribe(self, agent_id: str, message_type: MessageType):
        self.channel.subscribe(agent_id, message_type)

    def unsubscribe(self, agent_id: str, message_type: MessageType):
        self.channel.unsubscribe(agent_id, message_type)