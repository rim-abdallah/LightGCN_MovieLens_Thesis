def get_fixed_sample(data_generator, num_users, num_items):
    sampled_users = list(data_generator.train_items.keys())[:num_users]

    sampled_items = set()
    for u in sampled_users:
        sampled_items.update(data_generator.train_items[u][:5])

    sampled_items = list(sampled_items)[:num_items]

    return sampled_users, sampled_items