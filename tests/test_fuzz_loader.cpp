// Fuzz test of the model loader: damaged and hostile model files must give a namturbo::Error or a working model, and nothing else
// (no crash, no other exception, no hang, no huge allocation). The mutations are generated from a fixed seed, so a failure repeats.
// Usage: test_fuzz_loader <slimmable A2 model.nam> [iterations per kind (default 300)] [seed (default 1)]
// Exit codes: 0 pass, 1 failure, 77 skipped (model missing).
// A longer run under a sanitizer (see the build instructions) is how the loader is checked before a release; src/ has no
// parser of its own: the JSON parser is the vendored nlohmann/json.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "json.hpp"
#include "namturbo/namturbo.h"

using namespace namturbo;
using json = nlohmann::json;

namespace
{
int g_failures = 0;
std::map<std::string, int> g_outcomes;
int g_accepted_run = 0;
bool g_can_run = false;

const char* code_name(ErrorCode c)
{
  switch (c)
  {
    case ErrorCode::Io: return "Io";
    case ErrorCode::InvalidFile: return "InvalidFile";
    case ErrorCode::NotA2: return "NotA2";
    case ErrorCode::UnsupportedShape: return "UnsupportedShape";
    case ErrorCode::CpuUnsupported: return "CpuUnsupported";
    case ErrorCode::InvalidArgument: return "InvalidArgument";
    case ErrorCode::Unsupported: return "Unsupported";
  }
  return "?";
}

// Loads the text. Anything but a model or an Error is a failure; `what` says which input it was.
void try_load(const std::string& text, const std::string& what)
{
  try
  {
    auto model = Model::load(text.data(), text.size());
    ++g_outcomes["loaded"];
    if (g_can_run && g_accepted_run < 40)
    {
      ++g_accepted_run;
      for (Variant v : {Variant::Full, Variant::Lite})
        if (model->has(v))
        {
          Processor p(model);
          ProcessorConfig c;
          c.max_block = 64;
          c.variant = v;
          p.prepare(c);
          std::vector<float> in(256, 0.1f), out(256);
          p.process(in.data(), out.data(), 256); // the values may be anything for odd weights; it must not crash
        }
    }
  }
  catch (const Error& e)
  {
    ++g_outcomes[code_name(e.code())];
  }
  catch (const std::exception& e)
  {
    std::printf("  FAIL (%s): an exception that is not a namturbo::Error: %s\n", what.c_str(), e.what());
    ++g_failures;
  }
  catch (...)
  {
    std::printf("  FAIL (%s): an unknown exception\n", what.c_str());
    ++g_failures;
  }
}

// ---- mutations of the bytes ------------------------------------------------------------------------------------------------------
std::string mutate_bytes(const std::string& seed, std::mt19937_64& rng)
{
  std::string s = seed;
  auto pick = [&](size_t n) { return size_t(rng() % std::max<size_t>(n, 1)); };
  switch (rng() % 6)
  {
    case 0: // flip a few bytes
      for (int k = 0, n = 1 + int(rng() % 8); k < n && !s.empty(); ++k)
        s[pick(s.size())] = char(rng());
      break;
    case 1: // cut the end off
      s.resize(pick(s.size()));
      break;
    case 2: // delete a piece
    {
      const size_t at = pick(s.size()), len = pick(2000);
      s.erase(at, len);
      break;
    }
    case 3: // duplicate a piece
    {
      const size_t at = pick(s.size()), len = std::min<size_t>(pick(3000), s.size() - at);
      s.insert(pick(s.size()), s.substr(at, len));
      break;
    }
    case 4: // overwrite digits near the start (the configuration), where it matters most
      for (int k = 0, n = 1 + int(rng() % 4); k < n; ++k)
      {
        const size_t at = pick(std::min<size_t>(s.size(), 4000));
        if (at < s.size() && std::isdigit(static_cast<unsigned char>(s[at])))
          s[at] = char('0' + rng() % 10);
        else if (at < s.size())
          s[at] = "{}[],:\"-.e0"[rng() % 11];
      }
      break;
    default: // random bytes of random length at the start
    {
      std::string junk(pick(64), '\0');
      for (char& ch : junk)
        ch = char(rng());
      s = junk + s.substr(pick(s.size()));
    }
  }
  return s;
}

// ---- mutations of the structure --------------------------------------------------------------------------------------------------
void collect(json& node, std::vector<json*>& out, bool in_weights)
{
  if (!in_weights)
    out.push_back(&node);
  if (node.is_object())
  {
    for (auto& [key, value] : node.items())
      collect(value, out, in_weights || key == "weights");
  }
  else if (node.is_array() && !in_weights)
    for (auto& value : node)
      collect(value, out, in_weights);
}

const char* kInfMarker = "\"@@INF@@\"";

json random_value(std::mt19937_64& rng)
{
  switch (rng() % 14)
  {
    case 0: return nullptr;
    case 1: return -1;
    case 2: return 0;
    case 3: return 1;
    case 4: return 2147483647;
    case 5: return 4294967296.0;
    case 6: return 1e300;
    case 7: return -1e-300;
    case 8: return "text";
    case 9: return json::array();
    case 10: return json::object();
    case 11: return true;
    case 12: return 0.5;
    default: return 7 + int(rng() % 4);
  }
}

std::string mutate_structure(const json& seed, std::mt19937_64& rng)
{
  json doc = seed;
  std::vector<json*> nodes;
  collect(doc, nodes, false);
  json* target = nodes[rng() % nodes.size()];
  std::string text;
  switch (rng() % 5)
  {
    case 0: *target = random_value(rng); break;
    case 1: // delete a key or an element
      if (target->is_object() && !target->empty())
        target->erase(std::next(target->begin(), long(rng() % target->size())).key());
      else if (target->is_array() && !target->empty())
        target->erase(std::next(target->begin(), long(rng() % target->size())));
      else
        *target = random_value(rng);
      break;
    case 2: // the weights: wrong length or odd values
    {
      json* w = nullptr;
      for (json* n : nodes)
        if (n->is_object() && n->contains("weights") && (*n)["weights"].is_array())
          w = &(*n)["weights"];
      if (!w)
        break;
      switch (rng() % 5)
      {
        case 0: w->erase(std::next(w->begin(), long(rng() % (w->size() + 1) / 2 + 1)), w->end()); break;
        case 1: for (int k = 0; k < 1 + int(rng() % 500); ++k) w->push_back(0.25); break;
        case 2: (*w)[rng() % w->size()] = 1e39; break;       // too large for a float
        case 3: (*w)[rng() % w->size()] = kInfMarker; break; // replaced by 1e999 in the text
        default: (*w)[rng() % w->size()] = random_value(rng); break;
      }
      break;
    }
    case 3: // change a number a little
      if (target->is_number_integer())
        *target = target->get<long long>() + long(rng() % 5) - 2;
      else if (target->is_number_float())
        *target = target->get<double>() * (1.0 + double(long(rng() % 3) - 1));
      else
        *target = random_value(rng);
      break;
    default: // swap two values
    {
      json* other = nodes[rng() % nodes.size()];
      if (other != target && !(other->is_structured() && target->is_structured()))
        std::swap(*other, *target);
      else
        *target = random_value(rng);
    }
  }
  text = doc.dump();
  for (size_t at; (at = text.find(kInfMarker)) != std::string::npos;)
    text.replace(at, std::char_traits<char>::length(kInfMarker), "1e999");
  return text;
}

// ---- hostile shapes ---------------------------------------------------------------------------------------------------------------
void hostile_inputs(const std::string& seed)
{
  std::printf("hostile_inputs\n");
  try_load("", "empty");
  try_load(" \n\t ", "blanks");
  try_load("null", "null");
  try_load("[]", "array");
  try_load("{}", "empty object");
  try_load(std::string(5000, '\0'), "zero bytes");
  try_load(std::string(200000, '['), "200000 nested arrays");
  try_load(std::string(200000, '[') + std::string(200000, ']'), "200000 nested arrays, closed");
  try_load("{\"a\":" + std::string(100000, '{') , "nested objects, open");
  try_load("{\"version\":\"0.5.4\",\"architecture\":\"WaveNet\",\"config\":{},\"weights\":[]}", "no layers");
  try_load("{\"architecture\":\"SlimmableContainer\",\"config\":{\"submodels\":[]},\"weights\":[]}", "no submodels");
  try_load("{\"architecture\":\"SlimmableContainer\",\"config\":{\"submodels\":" + std::string(50000, '[') + "}}", "deep submodels");
  try_load("{\"architecture\":\"WaveNet\",\"config\":{\"layers\":[{\"channels\":1000000,\"dilations\":[1,2,4]}]},\"weights\":[1,2,3]}",
           "huge channels");
  try_load("{\"architecture\":\"WaveNet\",\"config\":{\"layers\":[{\"channels\":-8}]},\"weights\":[]}", "negative channels");
  try_load("{\"architecture\":\"WaveNet\",\"sample_rate\":-5,\"config\":{},\"weights\":[]}", "negative rate");
  try_load("\xEF\xBB\xBF" + seed, "byte order mark before a valid file");
  try_load(seed + std::string(100, ' ') + "garbage", "text after a valid file");
  try_load(seed.substr(0, seed.size() / 2), "half a file");
  try_load(seed.substr(0, seed.size() - 1), "one byte missing");
  // A weight that is infinite, too large for a float, or not a number must be refused, not run.
  {
    json plain = json::parse(json::parse(seed).at("config").at("submodels")[0].at("model").dump());
    for (const char* bad : {"1e999", "1e39", "-1e39"})
    {
      json copy = plain;
      copy["weights"][copy["weights"].size() / 2] = "@@BAD@@";
      std::string text = copy.dump();
      const size_t at = text.find("\"@@BAD@@\"");
      text.replace(at, 9, bad); // the nine characters of "@@BAD@@" with its quotes
      try
      {
        Model::load(text.data(), text.size());
        std::printf("  FAIL: a weight of %s was accepted\n", bad);
        ++g_failures;
      }
      catch (const Error& e)
      {
        if (e.code() != ErrorCode::InvalidFile)
        {
          std::printf("  FAIL: a weight of %s gave %s\n", bad, code_name(e.code()));
          ++g_failures;
        }
      }
    }
  }
  const void* null = nullptr;
  try
  {
    Model::load(null, 5);
    std::printf("  FAIL: a null pointer was accepted\n");
    ++g_failures;
  }
  catch (const Error& e)
  {
    if (e.code() != ErrorCode::InvalidArgument)
    {
      std::printf("  FAIL: null pointer gave %s\n", code_name(e.code()));
      ++g_failures;
    }
  }
  try
  {
    Model::load("a", 0);
  }
  catch (const Error&)
  {
  }
}
} // namespace

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2 || !std::ifstream(argv[1]).good())
  {
    std::printf("SKIP: model file not found\n");
    return 77;
  }
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 300;
  const unsigned long long seed_value = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1;
  g_can_run = check_cpu().supported;
  std::ifstream is(argv[1], std::ios::binary);
  std::stringstream ss;
  ss << is.rdbuf();
  const std::string seed_text = ss.str();
  json seed_json = json::parse(seed_text);
  // A second seed: one submodel on its own, a plain (not slimmable) model file.
  const std::string plain_text = seed_json.at("config").at("submodels")[0].at("model").dump();
  json plain_json = json::parse(plain_text);

  const auto t0 = std::chrono::steady_clock::now();
  std::mt19937_64 rng(seed_value);
  hostile_inputs(seed_text);

  std::printf("byte mutations of the slimmable file and the plain file, %d each\n", iterations);
  for (int i = 0; i < iterations; ++i)
  {
    try_load(mutate_bytes(seed_text, rng), "bytes of the slimmable file, iteration " + std::to_string(i));
    try_load(mutate_bytes(plain_text, rng), "bytes of the plain file, iteration " + std::to_string(i));
  }
  std::printf("structure mutations, %d each\n", iterations);
  for (int i = 0; i < iterations * 3; ++i)
  {
    try_load(mutate_structure(seed_json, rng), "structure of the slimmable file, iteration " + std::to_string(i));
    try_load(mutate_structure(plain_json, rng), "structure of the plain file, iteration " + std::to_string(i));
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("outcomes (seed %llu, %.1f s):", seed_value, seconds);
  for (const auto& [name, count] : g_outcomes)
    std::printf("  %s %d", name.c_str(), count);
  std::printf("\n");
  // The seed files themselves must load, and a mutation test that rejects everything proves nothing.
  try_load(seed_text, "the seed");
  try_load(plain_text, "the plain seed");
  if (g_outcomes["loaded"] < 2)
  {
    std::printf("  FAIL: the seed files did not load\n");
    ++g_failures;
  }
  std::printf(g_failures ? "FAILED: %d input(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
