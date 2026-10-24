-- Turn ::: {.more title="..."} blocks into collapsible <details>; leave other card divs as styled divs.
function Div(el)
  if el.classes:includes('more') then
    local title = el.attributes['title'] or 'Details'
    local out = {pandoc.RawBlock('html', '<details class="more"><summary>' .. title .. '</summary><div class="more-body">')}
    for _, block in ipairs(el.content) do table.insert(out, block) end
    table.insert(out, pandoc.RawBlock('html', '</div></details>'))
    return out
  end
end
