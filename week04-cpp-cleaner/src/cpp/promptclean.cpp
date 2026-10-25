// promptclean's compiled core: the same cleaning rules as src/promptclean/reference.py, in ONE pass over the
// UTF-8 bytes. The Python reference defines correct behaviour; tests compare the two on random Unicode.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // std::optional <-> None

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

// Rule 1: invisible format characters to delete (the zero-width joiner U+200D is deliberately kept).
bool is_invisible(char32_t cp) {
    return cp == 0x200B || cp == 0x200C || cp == 0x2060 || cp == 0xFEFF || cp == 0x00AD ||
           (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2066 && cp <= 0x2069);
}

// Rule 2: control characters to delete (tab and newline are kept: rules 3 and 4 handle them).
bool is_control(char32_t cp) {
    return (cp < 0x20 && cp != '\t' && cp != '\n') || cp == 0x7F || (cp >= 0x80 && cp <= 0x9F);
}

// Rule 3: horizontal whitespace, collapsed to one plain space.
bool is_horizontal_space(char32_t cp) { return cp == ' ' || cp == '\t' || cp == 0x00A0; }

// Decode one character starting at s[i]. Python always hands us valid UTF-8, so no error handling is needed.
// UTF-8 uses 1 to 4 bytes per character; the first byte says how many.
char32_t decode(const std::string& s, std::size_t i, std::size_t& len) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    auto cont = [&](std::size_t k) { return static_cast<char32_t>(static_cast<unsigned char>(s[i + k]) & 0x3F); };
    if (b0 < 0x80) { len = 1; return b0; }                                                  // 0xxxxxxx
    if (b0 < 0xE0) { len = 2; return (char32_t(b0 & 0x1F) << 6) | cont(1); }                // 110xxxxx 10xxxxxx
    if (b0 < 0xF0) { len = 3; return (char32_t(b0 & 0x0F) << 12) | (cont(1) << 6) | cont(2); }
    len = 4;
    return (char32_t(b0 & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3);
}

}  // namespace

std::string clean(const std::string& text, std::optional<long long> max_chars) {
    if (max_chars && *max_chars < 0) throw std::invalid_argument("max_chars must be >= 0");  // -> Python ValueError
    const bool capped = max_chars.has_value();
    const auto limit = capped ? static_cast<std::size_t>(*max_chars) : 0;

    std::string out;
    out.reserve(text.size());   // the result is never longer than the input: one allocation
    std::string pending;        // collapsed whitespace seen since the last real character: only " " and "\n"
    std::size_t out_chars = 0;  // characters (not bytes) written so far, for max_chars

    for (std::size_t i = 0; i < text.size();) {
        std::size_t len = 0;
        const char32_t cp = decode(text, i, len);
        const std::size_t start = i;
        i += len;

        if (is_invisible(cp) || is_control(cp)) continue;  // rules 1, 2: deleted; they don't break whitespace runs

        if (is_horizontal_space(cp)) {  // rule 3: a run of any length becomes one space
            if (pending.empty() || pending.back() != ' ') pending.push_back(' ');
            continue;
        }
        if (cp == '\n') {  // rule 4: a run of newlines keeps at most two
            const std::size_t n = pending.size();
            if (!(n >= 2 && pending[n - 1] == '\n' && pending[n - 2] == '\n')) pending.push_back('\n');
            continue;
        }

        // A real character. Whitespace before the FIRST one is dropped (rule 5, leading strip).
        if (out.empty()) pending.clear();
        if (capped && out_chars + pending.size() + 1 > limit) break;  // rule 6: the cap is reached. Whatever is
        // left of `pending` would be stripped as trailing whitespace anyway, so stopping here is exact.
        out += pending;
        out_chars += pending.size();
        pending.clear();
        out.append(text, start, len);  // copy the character's bytes unchanged
        ++out_chars;
    }
    // Whitespace after the last real character is never written (rule 5, trailing strip).
    return out;
}

// Step 2's pipeline check, kept as a tiny demonstration of bytes vs characters.
std::size_t utf8_bytes(const std::string& text) { return text.size(); }

PYBIND11_MODULE(_native, m) {
    m.doc() = "promptclean's compiled core";
    // call_guard<gil_scoped_release>: release the GIL while the C++ runs, so other Python threads (and the
    // event loop, when this is called via asyncio.to_thread) keep running. Safe here because pybind11 converts
    // the arguments BEFORE the guard (the input is already a C++ std::string copy), converts the result AFTER it,
    // and clean() touches no Python object in between. Touching one without the GIL would be a crash.
    m.def("clean", &clean, py::arg("text"), py::arg("max_chars") = py::none(),
          py::call_guard<py::gil_scoped_release>(),
          "Clean a prompt (same rules as promptclean.reference.clean), in one pass over the UTF-8 bytes. "
          "Releases the GIL while it works.");
    // The same function WITHOUT releasing the GIL. Kept only to demonstrate and test the difference
    // (scripts/gil_demo.py, tests/test_gil.py). Don't use it.
    m.def("clean_holding_gil", &clean, py::arg("text"), py::arg("max_chars") = py::none(),
          "Demo only: clean() without releasing the GIL");
    m.def("utf8_bytes", &utf8_bytes, "Length of the text in UTF-8 bytes (Python's len() counts characters)");
}
