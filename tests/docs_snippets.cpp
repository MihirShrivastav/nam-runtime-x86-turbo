// The code fragments of README.md and docs/integration-guide.md, so that the build fails if the documentation and the interface
// drift apart. This file is only compiled (an object library), never run: the fragments need a model file and a host.
#include <memory>
#include <string>
#include <utility>

#include "namturbo/namturbo.h"

void show_message(const std::string&);
void setLatencySamples(int);
double host_sample_rate();
int host_max_block();

// README, "Quick start".
void readme_quick_start(const float* in, float* out, int n)
{
  auto model = namturbo::Model::load("amp.nam");

  namturbo::Processor processor(model);
  namturbo::ProcessorConfig config;
  config.sample_rate = 48000.0;
  config.max_block = 256;
  processor.prepare(config);

  processor.process(in, out, n);
}

// Integration guide, sections 3 to 6.
void guide_check_and_load(const std::string& path)
{
  const namturbo::CpuSupport cpu = namturbo::check_cpu();
  if (!cpu.supported)
    show_message(cpu.detail);

  std::shared_ptr<const namturbo::Model> model;
  try
  {
    model = namturbo::Model::load(path);
  }
  catch (const namturbo::Error& e)
  {
    show_message(e.what());
    if (e.code() == namturbo::ErrorCode::NotA2)
      return;
  }

  namturbo::Processor processor(model);

  namturbo::ProcessorConfig config;
  config.sample_rate = host_sample_rate();
  config.max_block = host_max_block();
  config.variant = namturbo::Variant::Full;
  config.resampler = namturbo::ResamplerQuality::LowLatency;
  config.oversampling = namturbo::Oversampling::X2;
  config.monitoring = true;
  config.swap_fade_ms = 10.0;
  processor.prepare(config);

  float input[64] = {}, output[64];
  double dinput[64] = {}, doutput[64];
  processor.process(input, output, 64);
  processor.process(dinput, doutput, 64);
  processor.process(input, input, 64);

  setLatencySamples(processor.latency_samples());
  processor.reset();
  const namturbo::ModelInfo& info = model->info();
  (void)info.sample_rate;
  (void)info.loudness_db;
  if (auto db = namturbo::input_calibration_db(info, 12.0))
    (void)namturbo::db_to_linear(*db);
  if (auto db = namturbo::output_normalization_db(info))
    (void)namturbo::db_to_linear(*db);
  (void)model->has(namturbo::Variant::Lite);
}

// Integration guide, section 8: swap.
void guide_swap(namturbo::Processor& processor, std::shared_ptr<const namturbo::Model> new_model)
{
  namturbo::SwapTicket ticket = processor.prepare_swap(new_model, namturbo::Variant::Full);
  if (!processor.commit_swap(std::move(ticket)))
  {
    // the ticket was out of date
  }
  processor.service();
  (void)processor.swap_busy();
  (void)processor.swaps_completed();
  (void)processor.swaps_dropped();
}

// Integration guide, section 9: adaptive mode.
void guide_adaptive(namturbo::Processor& processor)
{
  namturbo::AdaptiveConfig adaptive;
  adaptive.enabled = true;
  adaptive.high_load = 0.70;
  adaptive.down_after_ms = 200;
  adaptive.low_load = 0.30;
  adaptive.up_after_ms = 4000;
  adaptive.on_switch = [](namturbo::Variant v) { (void)v; };
  processor.set_adaptive(adaptive);
  (void)processor.smoothed_load();
}

// Integration guide, section 10: monitoring.
void guide_monitoring(namturbo::Processor& processor)
{
  namturbo::PerfStats s = processor.stats();
  (void)s.blocks;
  (void)s.mean_us;
  (void)s.max_us;
  (void)s.mean_load;
  (void)s.p99_load;
  (void)s.max_load;
  (void)s.over_deadline;
  processor.reset_stats();
}

// The overview comment of the umbrella header.
void umbrella_comment(const float* in, float* out, int n)
{
  auto model = namturbo::Model::load("amp.nam");
  namturbo::Processor processor(model);
  namturbo::ProcessorConfig config;
  config.max_block = 512;
  processor.prepare(config);
  processor.process(in, out, n);
  (void)namturbo::version_string();
  (void)namturbo::version().major;
}
