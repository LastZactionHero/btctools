import json
import copy
import random

class GeneticContextModifier():
    CONTEXT_PATH = "./mutated_contexts"
    MUTATION_PERCENT = 0.03 # Any context parameter can be mutated +/- 5% 
    BASE_CONTEXT = {
        "raise_stoploss_threshold": 1.018,
        "sell_stoploss_floor": 0.002,
        "stop_loss_percent": 0.08,
        "max_delta": 4.5,
        "max_spread": 0.050,
    }

    def __init__(self):
        self.base_context = self.load_latest_context()
        self.trial_context = None

    # Loads the latest context file, or returns the base config.
    def load_latest_context(self):
        import os
        files = os.listdir(self.CONTEXT_PATH)
        if len(files) > 0:
            latest_file = sorted(files)[-1]
            with open(os.path.join(self.CONTEXT_PATH, latest_file), 'r') as f:
                return json.load(f)
        else:
            return {
                "total_net": None,
                "percent_change": None,
                "context": self.BASE_CONTEXT
            }

    # Randomly mtuates a single parameter of the context.
    def mutate_context(self):
        self.trial_context = copy.deepcopy(self.base_context)
        self.trial_context['total_net'] = None
        self.trial_context['percent_change'] = None
    
        key = random.choice(list(self.trial_context['context'].keys()))
        value = self.trial_context['context'][key]
        mutation = random.uniform(-self.MUTATION_PERCENT, self.MUTATION_PERCENT)
        self.trial_context['context'][key] = value * (1 + mutation)
        self.trial_context['total_net'] = None
        self.trial_context['percent_change'] = None
     
        print("Base Context")
        print(json.dumps(self.base_context, indent=2))
    
        print("New Context")
        print(json.dumps(self.trial_context, indent=2))
    
    # Saves a context file as JSON, with an increasing file number.
    def save_context(self, context):
        import os, json
        files = os.listdir(self.CONTEXT_PATH)
        if len(files) > 0:
            latest_number = int(sorted(files)[-1].split('.')[0].split('_')[-1])
        else:
            latest_number = 0
        new_file = os.path.join(self.CONTEXT_PATH, f"context_{latest_number + 1:04d}.json")
        with open(new_file, 'w') as f:
            json.dump(context, f)

    # Compares the base and trial contexts, and promotes the "trial" as "base" if it performs better.
    def compare_contexts(self, total_net, percent_change):
        self.trial_context['total_net'] = total_net
        self.trial_context['percent_change'] = percent_change

        # Both are good metrics for comparison
        net_is_better = self.trial_context['total_net'] >= self.base_context['total_net']
        percent_is_better = self.trial_context['percent_change'] >= self.base_context['percent_change']
        mutation_is_better =  net_is_better and percent_is_better

        if mutation_is_better:
            self.base_context = copy.deepcopy(self.trial_context)
            self.save_context(self.trial_context)
            print("Mutation is better!")
            return True
        print("Base is better!")
        return False