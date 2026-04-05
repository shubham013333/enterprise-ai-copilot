chat_history = []


def add_to_memory(user, ai):
    if user and ai:
        chat_history.append({
            "user": user,
            "ai": ai
        })


def get_memory():
    return chat_history


def clear_memory():
    chat_history.clear()