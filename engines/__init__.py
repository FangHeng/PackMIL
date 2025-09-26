from .base_engine import BaseTrainer
from .common_mil import CommonMIL

def build_engine(args,device,dataset):

    _commom_mil = ('mhim','mhim_pure','rrtmil','abmil','gattmil','clam_sb','clam_mb','transmil','dsmil','dtfd','meanmil','maxmil','vitmil','abmilx','wikg','gigap','chief','pack_pure','2dmamba')
    
    if args.model in _commom_mil:
        engine = CommonMIL(args)
    else:
        raise NotImplementedError
    trainer = BaseTrainer(engine=engine,args=args)

    if args.datasets.lower().startswith('surv'):
        return trainer.surv_train,trainer.surv_validate
    else:
        return trainer.train,trainer.validate
