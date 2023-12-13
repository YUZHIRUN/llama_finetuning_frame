import time


# def init_print():
#     for i in range(150):
#         if i % 2 == 0:
#             print('+', end='')
#             time.sleep(0.025)
#         else:
#             print('-', end='')
#         time.sleep(0.025)


def print_mention(content, rank=0, color='default'):
    if color == 'red':
        content = '\033[1;31m{}\033[0m'.format(content)
    elif color == 'green':
        content = '\033[1;32m{}\033[0m'.format(content)
    elif color == 'yellow':
        content = '\033[1;33m{}\033[0m'.format(content)
    elif color == 'blue':
        content = '\033[1;34m{}\033[0m'.format(content)
    else:
        content = content
    content = '-----------------------------> Info: {}'.format(content)
    if rank == 0:
        print(content)


def print_warning(content, rank=0, color='yellow'):
    if color == 'red':
        content = '\033[1;31m{}\033[0m'.format(content)
    elif color == 'green':
        content = '\033[1;32m{}\033[0m'.format(content)
    elif color == 'yellow':
        content = '\033[1;33m{}\033[0m'.format(content)
    elif color == 'blue':
        content = '\033[1;34m{}\033[0m'.format(content)
    else:
        content = content
    content = '-----------------------------! Warning: {}'.format(content)
    if rank == 0:
        print(content)


def print_arrow(content, color='green'):
    if color == 'red':
        content = '\033[1;31m{}\033[0m'.format(content)
    elif color == 'green':
        content = '\033[1;32m{}\033[0m'.format(content)
    elif color == 'yellow':
        content = '\033[1;33m{}\033[0m'.format(content)
    elif color == 'blue':
        content = '\033[1;34m{}\033[0m'.format(content)
    else:
        content = content
    content = f'-------> {content}'
    print(content)


def print_result(result: dict, rank):
    print_mention('Train has been completed!', rank, color='green')
    for key, value in result.items():
        if rank == 0:
            print_arrow(f'{key}: {value}')
    print_mention('END', color='green', rank=rank)


def update_kwargs(config, **kwargs):
    if isinstance(config, tuple):
        for config_item in config:
            update_kwargs(config_item, **kwargs)
    for key, value in kwargs.items():
        if '.' in key:
            father, son = key.split('.')
            if type(config).__name__ == father:
                if hasattr(config, son):
                    setattr(config, son, value)
        if hasattr(config, key):
            setattr(config, key, value)
