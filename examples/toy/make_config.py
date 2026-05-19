from toy_model import make_config

if __name__ == "__main__":
    cfg = make_config("toy_config.pkl")
    print("wrote toy_config.pkl")
    print("analytic posterior mean:", cfg.posterior_mean())
