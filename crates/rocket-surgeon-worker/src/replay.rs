use rocket_surgeon_protocol::messages::{Divergence, HostReplayRequest, ReplayStopAt};
use rocket_surgeon_protocol::types::InterventionRecipe;

#[allow(dead_code)]
pub struct ReplayContext {
    pub verify: bool,
    pub deterministic: bool,
    pub cosine_threshold: f64,
    pub mre_threshold: f64,
    pub stop_at: Option<ReplayStopAt>,
    pub interventions: Vec<InterventionRecipe>,
    pub divergences: Vec<Divergence>,
    pub ticks_replayed: u32,
}

impl ReplayContext {
    pub fn from_request(req: &HostReplayRequest) -> Self {
        Self {
            verify: req.verify,
            deterministic: req.deterministic,
            cosine_threshold: req.cosine_threshold,
            mre_threshold: req.mre_threshold,
            stop_at: req.stop_at.clone(),
            interventions: req.interventions.clone(),
            divergences: Vec::new(),
            ticks_replayed: 0,
        }
    }

    pub fn should_stop(&self, layer: u32, component: &str) -> bool {
        if let Some(ref stop) = self.stop_at {
            layer == stop.layer && component == stop.component
        } else {
            false
        }
    }
}
