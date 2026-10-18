"""Content-free native health instrumentation for the pinned backend."""

HELPER = r'''
// Operational metadata only: no exception messages, requests or model values.
static void nla_native_health(const char * category, const char * file,
                              const char * function, int line, const char * channel) noexcept {
    try {
        const char * directory = std::getenv("NLA_NATIVE_HEALTH_DIR");
        const char * job = std::getenv("LSB_JOBID");
        const char * role = std::getenv("NLA_NATIVE_ROLE");
        if (!directory || !job || !role) return;
        static std::mutex health_mutex;
        std::lock_guard<std::mutex> guard(health_mutex);
        std::string filename(file);
        filename = filename.substr(filename.find_last_of("/\\") + 1);
        const std::string path = std::string(directory) + "/native-health-" + job + "-" + role + "-" + channel + ".json";
        const std::string temporary = path + ".tmp";
        const json value = {{"error_type", category}, {"frames", json::array({
            {{"file", filename}, {"function", function}, {"line", line}}
        })}};
        std::ofstream output(temporary);
        output << value.dump();
        output.close();
        if (output) std::rename(temporary.c_str(), path.c_str());
    } catch (...) {} // Diagnostics must never affect inference or error handling.
}

'''

RUNTIME_HELPER = r'''
static std::runtime_error nla_native_runtime_error(const std::string & message,
        const char * file = __builtin_FILE(), const char * function = __builtin_FUNCTION(),
        int line = __builtin_LINE()) {
    nla_native_health("std::runtime_error", file, function, line, "throw");
    return std::runtime_error(message);
}

'''

def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected pinned native health source')
    return source.replace(old, new, 1)

def instrument(source, kind, runtime_frames=True):
    source = '#include <fstream>\n#include <cstdlib>\n#include <cstdio>\n#include <mutex>\n#include <typeinfo>\n' + source
    markers = {'http': '// wrapper function that handles exceptions and logs errors',
               'context': '// fix problem with std::min and std::max',
               'common': 'json format_error_response(',
               'task': '//\n// task_params\n'}
    marker = markers[kind]
    source = replace_once(source, marker, HELPER + (RUNTIME_HELPER if runtime_frames and kind != 'http' else '') + marker)
    if kind == 'http':
        source = replace_once(source, '#include "server-context.h"', '#include "server-context.h"\n#include "server-common.h"')
        for declaration, category in [('const std::invalid_argument & e', 'typeid(e).name()'),
                                       ('const std::exception & e', 'typeid(e).name()')]:
            old = '} catch (' + declaration + ') {'
            # The wrapper's first matching catch is the relevant one.
            at = source.index(old)
            source = source[:at] + source[at:].replace(old, old + '\n            nla_native_health(' + category + ', __FILE__, "ex_wrapper", __LINE__, "http");', 1)
        source = replace_once(source, '            message = "unknown error";',
            '            nla_native_health("UnknownException", __FILE__, "ex_wrapper", __LINE__, "http");\n            message = "unknown error";')
    elif kind == 'context':
        for argument in ('const server_task & task', 'const server_slot & slot'):
            old = 'void send_error(' + argument + ', const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {'
            new = old[:-3] + ', const char * health_file = __builtin_FILE(), const char * health_function = __builtin_FUNCTION(), int health_line = __builtin_LINE()) {'
            source = replace_once(source, old, new)
        source = replace_once(source, 'send_error(task.id, error, type);', 'send_error(task.id, error, type, 0, 0, health_file, health_function, health_line);')
        source = replace_once(source, 'send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);', 'send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx, health_file, health_function, health_line);')
        old = 'const int32_t n_ctx = 0) {\n        SRV_ERR('
        new = 'const int32_t n_ctx = 0, const char * health_file = __builtin_FILE(), const char * health_function = __builtin_FUNCTION(), int health_line = __builtin_LINE()) {\n        nla_native_health("NativeTaskError", health_file, health_function, health_line, "context");\n        SRV_ERR('
        source = replace_once(source, old, new)
    if runtime_frames and kind != 'http':
        source = source.replace('throw std::runtime_error(', 'throw nla_native_runtime_error(')
    return source
